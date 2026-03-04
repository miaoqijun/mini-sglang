from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
    DebugInfoMsg
)
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.frontend import Node

class NodeAllFinished(Exception):
    pass


@dataclass
class NodeStatus:
    input_ids: List[int]
    output_ids: List[int]
    inherited_len: int = 0
    cached_len: int = -1

@dataclass
class NodeInfo:
    # inference info
    node_type: str
    name: str
    input_organization: List[PromptComponent]
    sampling_params: SamplingParams

    # DAG info
    in_degree: int
    successors: Set[int]


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

        self.debug = debug

        self.pending_nodes: List[int] = []
        self.status_map: Dict[int, NodeStatus] = {}
        self.info_map: Dict[int, NodeInfo] = {}
        self.sink_nodes: List[int] = []
        self.num_completed = 0

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def _get_node_prompts(self, node_id: str) -> Tuple(List[int], int):
        text = ""
        inherited_ids = []
        for i, prompt_component in enumerate(self.info_map[node_id].input_organization):
            if prompt_component.node_ref is None:
                text += prompt_component.text
            else:
                ref_status = self.status_map[prompt_component.node_ref]
                if prompt_component.text == "generated":
                    ref_ids = ref_status.output_ids
                else: # all
                    if i == 0: # first components and all reference, note that
                        inherited_ids = ref_status.input_ids + ref_status.output_ids
                    ref_ids = ref_status.input_ids + ref_status.output_ids
                text += self.tokenizer.decode(ref_ids)
        return text, len(inherited_ids)
    
    def _get_node_input_ids(self, node_id: str) -> List[int]:
        input_ids = []
        for prompt_component in self.info_map[node_id].input_organization:
            if prompt_component.node_ref is not None:
                ref_status = self.status_map[prompt_component.node_ref]
                if prompt_component.text == "generated":
                    input_ids += ref_status.output_ids
                else: # all
                    input_ids += ref_status.input_ids + ref_status.output_ids
            else:
                input_ids += self._tokenize_one(prompt_component.text)
        return input_ids

    def _prepare_dag(self, nodes: List[Node]) -> None:
        self.pending_nodes = []
        self.status_map = {}
        self.info_map = {}
        
        for node in nodes:
            self.info_map[node.uid] = NodeInfo(
                node_type=node.node_type, input_organization=node.inputs, sampling_params=node.sampling_params, name=node.name,
                in_degree=0, successors=set()
            )

        for node in nodes:
            for prompt_component in node.inputs:
                if prompt_component.node_ref is not None:
                    assert prompt_component.node_ref in self.info_map, f"Invalid node reference from {node.uid} to {prompt_component.node_ref}"
                    self.info_map[node.uid].in_degree += 1
                    self.info_map[prompt_component.node_ref].successors.add(node.uid)
        
        for uid, node_info in self.info_map.items():
            if node_info.in_degree == 0:
                self.pending_nodes.append(uid)
            if len(node_info.successors) == 0:
                self.sink_nodes.append(uid)
    
    def _process_last_data(self, last_data: ForwardData | None) -> None:
        """
        modified from original scheduler, for sending back cache hit tokens (for now)
        """
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        info: List[DebugInfoMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                # Added: send debug info
                if self.debug:
                    info.append(DebugInfoMsg(uid=req.uid, cached_len=req.cached_len))

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
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply, info)
    
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
        debug_info['hit_rate'] = total_cached / total_inherited
        # 2.... other debug info (to be implemented)
        return debug_info

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if self.num_completed == len(self.info_map):
            raise NodeAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for node_id in self.pending_nodes:
            # We defer the budget exceeded detection to prefill manager for fully cache-aware scheduling.
            # May cause too much CPU computation when many nodes are ready in the same time.
            # if sum_input_len >= self.prefill_budget:
            #     break
            prompt, inherited_len = self._get_node_prompts(node_id)
            input_ids = self._tokenize_one(prompt)
            sampling_params = self.info_map[node_id].sampling_params
            sum_input_len += len(input_ids)
            added += 1
            results.append(UserMsg(uid=node_id, input_ids=input_ids, sampling_params=sampling_params))
            self.status_map[node_id] = NodeStatus(
                input_ids=input_ids.tolist(),
                output_ids=[],
                inherited_len=inherited_len,
            )
        self.pending_nodes = self.pending_nodes[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg], info: List[DebugInfoMsg]) -> None:
        # process debug info
        for msg in info:
            status = self.status_map[msg.uid]
            if status.cached_len == -1: # when requests got chunked, only update cache_len for the first chunk
                status.cached_len = msg.cached_len

        for msg in reply:
            status = self.status_map[msg.uid]
            status.output_ids.append(msg.next_token)
            if msg.finished and msg.next_token == self.eos_token_id: # request end, update dag
                self.num_completed += 1
                free_nodes = [msg.uid]
                while len(free_nodes) > 0:
                    next_free_nodes = []
                    for free_node in free_nodes:
                        for successor in self.info_map[free_node].successors:
                            successor_info = self.info_map[successor]
                            successor_info.in_degree -= 1
                            if successor_info.in_degree == 0:
                                if successor_info.node_type == "inference":
                                    self.pending_nodes.append(successor)
                                elif successor_info.node_type == "concatenate":
                                    self.num_completed += 1
                                    input_ids = self._get_node_input_ids(successor)
                                    self.status_map[successor] = NodeStatus(
                                        input_ids=input_ids,
                                        output_ids=input_ids,
                                    )
                                    next_free_nodes.append(successor)
                    free_nodes = next_free_nodes

    def run_workflow(
        self,
        nodes: List[Node],
    ) -> Dict[int, Dict[str, str | List[int]]]:
        self._prepare_dag(nodes)
        try:
            self.run_forever()
        except NodeAllFinished:
            pass
        results: Dict[int, Dict[str, str | List[int]]] = {}
        for sink_node in self.sink_nodes:
            status = self.status_map[sink_node]
            output_text = self.tokenizer.decode(status.output_ids)
            results[sink_node] = {"text": output_text, "token_ids": status.output_ids}
        if self.debug:
            return results, self._get_debug_info()
        else:
            return results