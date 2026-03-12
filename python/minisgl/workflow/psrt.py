from typing import List, Tuple, Dict
from minisgl.frontend import Node

class PSRTNode:
    def __init__(self, node: Node):
        self.uid = node.uid
        self.name = node.name
        self.sampling_params = node.sampling_params
        if node.inputs[0].node_ref is not None and node.inputs[0].text == "all":
            self.parent = node.inputs[0].node_ref # store uid first, processed in get_PSRTs() later
        else:
            self.parent = None

        self.inputs = node.inputs
    
        self.in_degree = sum([int(input_component.node_ref is not None) for input_component in self.inputs])
        self.ready_predecessor = 0

        self.children = []
        self.successors = []

        self.root_uid = None

        self.t = None # if root, record the scheduling timestamp
        self.size = None
        self.completed_nodes = 0
    
    def get_root(self):
        if self.root_uid is None:
            node = self.parent
            while node.parent is not None:
                node = node.parent
            self.root_uid = node.uid
        return self.root_uid

    def get_size(self):
        if self.size is None:
            children_sizes = 0
            for child in self.children:
                children_sizes += child.get_size()
            self.size = children_sizes + 1
        return self.size

def get_PSRTs(nodes: List[Node]) -> Tuple[List[PSRTNode], Dict[int, PSRTNode]]: # return roots
    tnodes = [PSRTNode(node) for node in nodes]
    uid2node = {node.uid: node for node in tnodes}
    roots = []
    for node in tnodes:
        if node.parent is None:
            roots.append(node)
        else:
            node.parent = uid2node[node.parent]
            node.parent.children.append(node)
        for input_component in node.inputs:
            if input_component.node_ref is not None:
                predecessor = uid2node[input_component.node_ref]
                predecessor.successors.append(node)
    return roots, uid2node
