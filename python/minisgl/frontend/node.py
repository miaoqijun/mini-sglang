import uuid
from typing import List
from minisgl.core import SamplingParams

class PromptComponent:
    def __init__(self, text, node_ref=None):
        self.text = text
        self.node_ref = node_ref
        if node_ref is not None:
            assert self.text == "all" or self.text == "generated", "If `node_ref` specified, `text` must be \'all\' or \'generated\'."

class Node:
    def __init__(self, gid: int, inputs: List[PromptComponent], sampling_params: SamplingParams = None, name: str = None, node_type: str = "inference"):
        self.gid = gid
        self.uid = uuid.uuid4().int
        self.inputs = inputs
        self.name = name
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.sampling_params = sampling_params
        self.node_type = node_type