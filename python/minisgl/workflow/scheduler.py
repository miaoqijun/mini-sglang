from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
from minisgl.core import SamplingParams, Req
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg
)
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.frontend import Node
from minisgl.utils import init_logger
from tqdm import tqdm

from .psrt import PSRTNode, get_PSRTs
from .prefill import PSRTPrefillManager

logger = init_logger(__name__)

class NodeAllFinished(Exception):
    pass

@dataclass
class NodeStatus:
    input_ids: List[int]
    output_ids: List[int]
    inherited_len: int = 0
    cached_len: int = -1

class WorkflowScheduler(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, debug: bool = True, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.prefill_manager = PSRTPrefillManager(self.cache_manager, self.table_manager, self.decode_manager)

        self.debug = debug

        self.status_map: Dict[int, NodeStatus] = {}
        self.info_map: Dict[int, PSRTNode] = {}
        self.ready_nodes: List[PSRTNode] = []
        self.node_cnt = 0
        self.completed_node = set()
        self.scheduled_order = []

        self.t = 0
        self.pbar = None
        self.cur_inherited = 0
        self.cur_cached = 0

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def _get_node_prompts(self, node: PSRTNode) -> Tuple(List[int], int):
        text = ""
        inherited_ids = []
        for i, prompt_component in enumerate(node.inputs):
            if prompt_component.node_ref is None:
                text += prompt_component.text
            else:
                ref_status = self.status_map[prompt_component.node_ref]
                if prompt_component.text == "generated":
                    ref_ids = ref_status.output_ids
                else: # all
                    ref_ids = ref_status.input_ids + ref_status.output_ids
                    if i == 0: # first components and all reference, note that
                        inherited_ids = ref_ids                   
                text += self.tokenizer.decode(ref_ids)
        return text, len(inherited_ids)

    def _check_sampling_params(self, sampling_params, input_len):
        max_seq_len = self.engine.max_seq_len
        max_output_len = max_seq_len - input_len
        if max_output_len <= 0:
            return logger.warning_rank0(
                f"Input sequence length {input_len} exceeds {max_seq_len}, "
                f"request {node.uid} is dropped."
            )
        if sampling_params.max_tokens > max_output_len:
            sampling_params.max_tokens = max_output_len
            logger.warning_rank0(
                f"Adjust max_tokens to {max_output_len} for request {node.uid}."
            )
        return sampling_params
    
    def _add_requests(self):
        if len(self.completed_node) == self.node_cnt:
            raise NodeAllFinished()

        for node in self.ready_nodes:
            prompt, inherited_len = self._get_node_prompts(node)
            input_ids = self._tokenize_one(prompt)
            self.status_map[node.uid] = NodeStatus(
                input_ids=input_ids.tolist(),
                output_ids=[],
                inherited_len=inherited_len,
            )     

            sampling_params = self._check_sampling_params(node.sampling_params, len(input_ids))
            if node.parent is None: # record psrt enter time
                node.t = self.t
                self.t += 1
            self.prefill_manager.add_one_req(node.uid, input_ids, sampling_params, self.info_map[node.get_root()].t)
        self.ready_nodes = []
    
    def _spawn_children_reqs(self, parent_req: Req, node: PSRTNode) -> None:
        self.cache_manager.cache_req(parent_req, finished=False)
        for i, child in enumerate(node.children):
            child_handle = parent_req.cache_handle 
            if i == 0:
                # first child: inheret parent's table_idx and handle
                child_table_idx = parent_req.table_idx
                child_handle = parent_req.cache_handle 
            else:
                # next children: 
                # add node reference
                self.cache_manager.lock(child_handle)
                # need to allocate in PrefillAdder
                child_table_idx = None

            # prepare a new req
            assert len(child.inputs) == 2 and child.inputs[1].node_ref is None, "We are assuming only root nodes have inter-psrt dependencies."
            input_ids = torch.cat([parent_req.input_ids, self._tokenize_one(child.inputs[1].text)])
            self.status_map[child.uid] = NodeStatus(
                input_ids=input_ids.tolist(),
                output_ids=[],
                inherited_len=len(parent_req.input_ids),
            )    
            sampling_params = self._check_sampling_params(child.sampling_params, len(input_ids))
            child_req = Req(
                input_ids=input_ids,
                table_idx=child_table_idx,
                cached_len=len(parent_req.input_ids),
                output_len=sampling_params.max_tokens,
                uid=child.uid,
                sampling_params=parent_req.sampling_params, 
                cache_handle=child_handle,
            )
            self.prefill_manager.add_child_req(child_req, self.info_map[child.get_root()].t)

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)

                    # expand psrt or free (leave node)
                    # TODO: we are assuming only root nodes have inter-psrt dependencies, so children are ready as soon as their parents are completed
                    node = self.info_map[req.uid]
                    if len(node.children) > 0:
                        self._spawn_children_reqs(req, node)
                    else:
                        self._free_req_resources(req)

                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None and self.prefill_manager.runnable: # starvation
            logger.warning_rank0("Scheduling starved due to insufficient memory. Triggering eviction.")
            self.prefill_manager.evict_one()
        return self._prepare_batch(batch) if batch else None
    
    def _get_debug_info(self):
        debug_info = {}
        # 1. cache hit rate
        total_inherited = 0
        total_cached = 0
        for uid, node_status in self.status_map.items():
            if node_status.inherited_len > 0:
                assert node_status.cached_len >= 0, f"node {self.info_map[uid].name} not return a debug info message"
                total_inherited += node_status.inherited_len
                total_cached += min(node_status.cached_len, node_status.inherited_len) # only care tokens that should be cached
        debug_info['total_inherited'] = total_inherited
        debug_info['total_cached'] = total_cached
        if total_inherited > 0:
            debug_info['hit_rate'] = total_cached / total_inherited
        # ... other debug info (to be implemented)
        return debug_info

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            status.output_ids.append(msg.next_token)
            if msg.uid not in self.completed_node and msg.finished: # request end, update dag
                self.pbar.update(1)
                self.completed_node.add(msg.uid)

                for successor in self.info_map[msg.uid].successors:
                    if successor.parent is not None: # only need to wake up other PSRT's root
                        continue
                    successor.ready_predecessor += 1
                    if successor.ready_predecessor == successor.in_degree:
                        self.ready_nodes.append(successor)
    
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        self._add_requests()

        forward_input = self._schedule_next_batch()

        # modified: update cached len for debug before forward
        if self.debug and forward_input is not None:
            for req in forward_input.batch.reqs:
                status = self.status_map[req.uid]
                if status.cached_len == -1: # only update cache_len for the first time
                    status.cached_len = req.cached_len
                    self.cur_inherited += status.inherited_len
                    self.cur_cached += min(status.cached_len, status.inherited_len)
                    if self.cur_inherited > 0:
                        self.pbar.set_postfix(hit_rate=self.cur_cached / self.cur_inherited, refresh=True)

        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def run_workflow(
        self,
        nodes: List[Node],
    ) -> Dict[int, Dict[str, str | List[int]]]:
        self.node_cnt = len(nodes)
        self.pbar = tqdm(total=self.node_cnt, desc=f"running inference for {self.node_cnt} nodes")
        self.roots, self.info_map = get_PSRTs(nodes)
        self.ready_nodes = [root for root in self.roots if root.in_degree == 0]
        self.prefill_manager.waiting_queue = [root for root in self.roots]
        try:
            self.run_forever()
        except NodeAllFinished:
            pass
        results: Dict[int, Dict[str, str | List[int]]] = {}
        for node in nodes:
            status = self.status_map[node.uid]
            output_text = self.tokenizer.decode(status.output_ids)
            results[node.uid] = {"text": output_text, "token_ids": status.output_ids, "output_len": len(status.output_ids)}
        if self.debug:
            return results, self._get_debug_info()
        else:
            return results, None