from dataclasses import dataclass, field
from typing import List
import bisect

from minisgl.scheduler.prefill import ChunkedReq, PrefillAdder, PrefillManager
from minisgl.scheduler.utils import PendingReq
from minisgl.utils import init_logger
from minisgl.core import Batch, Req
from .psrt import PSRTNode

logger = init_logger(__name__)

@dataclass
class PSRTPrefillAdder(PrefillAdder):
    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            if chunked_req.table_idx is None:
                if self.table_manager.available_size == 0:
                    return None

                chunked_req.table_idx = self.table_manager.allocate()
                # set the inherited part
                indices = chunked_req.cache_handle.get_matched_indices()
                self.table_manager.page_table[chunked_req.table_idx][:len(indices)].copy_(indices)
                self.table_manager.token_pool[chunked_req.table_idx][:len(indices)].copy_(
                    chunked_req.input_ids[:len(indices)].pin_memory(), non_blocking=True
                )

            # access control for chunked_req (possibly a child)
            extend_len = pending_req.input_len - chunked_req.cached_len
            estimated_len = extend_len + pending_req.output_len
            if estimated_len + self.reserved_size > self.cache_manager.available_size:
                return None

            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None

@dataclass
class PSRTPrefillManager(PrefillManager):
    priority_list: List[int] = field(default_factory=list)

    def evict_one(self) -> bool:
        # find child requests from the end of the pending_list
        for i in range(len(self.pending_list) - 1, -1, -1):
            req = self.pending_list[i]
            if req.chunked_req is not None and type(req.chunked_req) is Req: # child req
                self.cache_manager.unlock(req.chunked_req.cache_handle)
                
                req.chunked_req = None 
                logger.info_rank0(f"Evicted pending child_req {req.uid} to free up space.")
                return True
        return False

    def add_one_req(self, uid, input_ids, sampling_params, priority) -> None:
        idx = bisect.bisect_right(self.priority_list, priority)
        self.priority_list.insert(idx, priority)
        self.pending_list.insert(idx, PendingReq(uid, input_ids, sampling_params))

    def add_child_req(self, req: Req, priority) -> None:
        pending = PendingReq(req.uid, req.input_ids, req.sampling_params)
        # disguise as chunked_req
        pending.chunked_req = req 
        idx = bisect.bisect_right(self.priority_list, priority)
        self.priority_list.insert(idx, priority)
        self.pending_list.insert(idx, pending)

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PSRTPrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        while True:
            added = 0
            budget_exhausted = False

            for pending_req in self.pending_list:
                if req := adder.try_add_one(pending_req):
                    pending_req.chunked_req = None
                    if isinstance(req, ChunkedReq):
                        pending_req.chunked_req = req
                        chunked_list.append(pending_req)
                    reqs.append(req)
                    added += 1
                else:
                    if adder.token_budget <= 0:
                        budget_exhausted = True
                    break  # We cannot add more requests

            self.pending_list = self.pending_list[added:]
            if budget_exhausted or len(self.pending_list) == 0:
                break
            if not self.evict_one():
                break
        self.pending_list = chunked_list + self.pending_list

        if len(reqs) == 0:
            return None
        return Batch(reqs=reqs, phase="prefill")
