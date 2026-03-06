from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg
)
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.frontend import Node
from tqdm import tqdm

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

    # debug info
    queue_no: int = 0


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
        self.completed_node = set()
        self.scheduled_order = []
        self.pbar = None

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
                    ref_ids = ref_status.input_ids + ref_status.output_ids
                    if i == 0: # first components and all reference, note that
                        inherited_ids = ref_ids                   
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
        # check unique id
        nodes_set = set([node.uid for node in nodes])
        assert len(nodes_set) == len(nodes), "Repeated node uid detected."

        self.pending_nodes = []
        self.status_map = {}
        self.info_map = {}
        
        inference_node_cnt = 0
        for node in nodes:
            if node.node_type == 'inference':
                inference_node_cnt += 1
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

        if inference_node_cnt > 0:
            self.pbar = tqdm(total=inference_node_cnt, desc=f"running inference for {inference_node_cnt} nodes")
    
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
        # 2. schedule order
        debug_info['scheduled_order'] = self.scheduled_order
        # ... other debug info (to be implemented)
        return debug_info

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and self.num_completed == len(self.info_map):
            raise NodeAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for node_id in self.pending_nodes:
            if sum_input_len >= self.prefill_budget:
                break
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
        self.scheduled_order += [self.info_map[uid].name for uid in self.pending_nodes[:added]]
        self.pending_nodes = self.pending_nodes[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            status.output_ids.append(msg.next_token)
            if msg.uid not in self.completed_node and msg.finished: # request end, update dag
                self.num_completed += 1
                self.pbar.update(1)
                self.completed_node.add(msg.uid)

                free_nodes = [msg.uid]
                while len(free_nodes) > 0:
                    next_free_nodes = []
                    for free_node in free_nodes:
                        for successor in self.info_map[free_node].successors:
                            successor_info = self.info_map[successor]
                            successor_info.in_degree -= 1
                            if successor_info.in_degree == 0:
                                if successor_info.node_type == "inference":
                                    self.info_map[successor].queue_no = len(self.pending_nodes) + len(self.prefill_manager.pending_list)
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
    
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()

        # modified: update cached len for debug before forward
        if self.debug and forward_input is not None:
            for req in forward_input.batch.reqs:
                status = self.status_map[req.uid]
                if status.cached_len == -1: # only update cache_len for the first time
                    status.cached_len = req.cached_len

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