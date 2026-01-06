from __future__ import annotations

import logging
from typing import Any
import onnx_ir as ir

def generate_tensor_name() -> str:
    # TODO: check if the name is already used
    global _tensor_counter
    _tensor_counter += 1
    return f"tensor_{_tensor_counter}"

def generate_node_name(op: str) -> str:
    # TODO: check if the name is already used
    global _node_counter
    _node_counter += 1
    return f"{op}_{_node_counter}"


class TensorMatcher:
    __slots__ = ("_name", "_producer", "_consumers", "_index")

    def __init__(
        self,
        name: str | None = None,
    ) -> None:
        self._name = name
        self._producer: NodeMatcher | None = None
        self._consumers: list[NodeMatcher] = []
        self._index: int | None = None

    @property
    def name(self) -> str | None:
        return self._name
    
    @property
    def producer(self) -> NodeMatcher | None:
        return self._producer

    @producer.setter
    def producer(self, producer: NodeMatcher | None) -> None:
        self._producer = producer
    
    @property
    def consumers(self) -> list[NodeMatcher]:
        return self._consumers
    
    @consumers.setter
    def consumers(self, consumers: list[NodeMatcher]) -> None:
        self._consumers = consumers

    def add_consumer(self, consumer: NodeMatcher) -> None:
        self._consumers.append(consumer)
    
    @property
    def index(self) -> int | None:
        return self._index
    
    @index.setter
    def index(self, index: int | None) -> None:
        self._index = index

    def __hash__(self) -> int:
        return hash((self._name))
    
    def __str__(self) -> str:
        return f"TensorMatcher(name={self._name}, producer={self._producer}, consumers={self._consumers}, index={self._index})"

    def __repr__(self) -> str:
        return f"TensorMatcher(name={self._name}, producer={self._producer}, consumers={self._consumers}, index={self._index})"


class NodeMatcher:
    __slots__ = ("_op", "_name", "_attrs", "_inputs", "_outputs")

    def __init__(
        self,
        op: str,
        name: str | None = None,
        attrs: dict[str, Any] | None = None,
        inputs: list[TensorMatcher] | None = None,
        outputs: list[TensorMatcher] | None = None,
    ) -> None:
        self._op = op
        self._name = name or generate_node_name(op)
        self._attrs = attrs or {}
        self._inputs = inputs or []
        self._outputs = outputs or [TensorMatcher(generate_tensor_name())]

        for idx, output in enumerate(self._outputs):
            output.producer = self
            output.index = idx
            output.add_consumer(self)
        
    @property
    def op(self) -> str:
        return self._op
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def attrs(self) -> dict[str, Any]:
        return self._attrs
    
    @property
    def inputs(self) -> list[TensorMatcher]:
        return self._inputs
    
    @property
    def outputs(self) -> list[TensorMatcher]:
        return self._outputs
    
    def __str__(self) -> str:
        return f"NodeMatcher(op={self._op}, name={self._name}, attrs={self._attrs}, inputs={self._inputs}, outputs={self._outputs})"
    
    def __repr__(self) -> str:
        return f"NodeMatcher(op={self._op}, name={self._name}, attrs={self._attrs}, inputs={self._inputs}, outputs={self._outputs})"

    def __hash__(self) -> int:
        return hash(f"{self._op}_{self._name}")

    def __call__(
        self,
        inputs: TensorMatcher | list[TensorMatcher],
    ) -> TensorMatcher | list[TensorMatcher]:
        self._inputs = inputs if isinstance(inputs, list) else [inputs]
        return self._outputs[0] if len(self._outputs) == 1 else self._outputs

    def _check_op_type(
        self,
        node: ir.Node
    ) -> bool:
        return self._op == node.op_type or self._op == "Wildcard"
    
    def match(
        self,
        node: ir.Node,
        matched_nodes: dict[NodeMatcher, ir.Node],
        matched_tensors: dict[TensorMatcher, ir.Tensor],
    ) -> bool:
        if not self._check_op_type(node):
            return False
        
        matched_nodes[self] = node

        for target_node_in, matcher_node_in in zip(node.inputs, self._inputs):
            matched_tensors[matcher_node_in] = target_node_in

            if matcher_node_in.producer is None:
                continue
            
            if matcher_node_in.producer in matched_nodes:
                continue
            
            if target_node_in.producer() is None:
                return False
            
            if not matcher_node_in.producer.match(target_node_in.producer(), matched_nodes, matched_tensors):
                return False
        
        return True


class GraphMatcher:
    __slots__ = ("_nodes", "_inputs", "_outputs", "_name")

    def __init__(
        self,
        nodes: list[NodeMatcher],
        inputs: list[TensorMatcher],
        outputs: list[TensorMatcher],
        name: str
    ) -> None:
        self._nodes = nodes
        self._inputs = inputs
        self._outputs = outputs
        self._name = name

    @property
    def nodes(self) -> list[NodeMatcher]:
        return self._nodes
    
    @property
    def inputs(self) -> list[TensorMatcher]:
        return self._inputs
    
    @property
    def outputs(self) -> list[TensorMatcher]:
        return self._outputs
    
    @property
    def name(self) -> str:
        return self._name
    
    def __str__(self) -> str:
        return f"GraphMatcher(nodes={self._nodes}, inputs={self._inputs}, outputs={self._outputs}, name={self._name})"
    
    def __repr__(self) -> str:
        return f"GraphMatcher(nodes={self._nodes}, inputs={self._inputs}, outputs={self._outputs}, name={self._name})"
    
    def __hash__(self) -> int:
        return hash(self._name)

    def match(
        self,
        graph: ir.Graph,
    ) -> tuple[list[dict[NodeMatcher, ir.Node]], list[dict[TensorMatcher, ir.Tensor]]]:
        matched_nodes_list: list[dict[NodeMatcher, ir.Node]] = []
        matched_tensors_list: list[dict[TensorMatcher, ir.Tensor]] = []

        root_matcher = self._nodes[-1]
        for node in graph:
            matched_nodes: dict[NodeMatcher, ir.Node] = {}
            matched_tensors: dict[TensorMatcher, ir.Tensor] = {}

            if root_matcher.match(node, matched_nodes, matched_tensors):
                root_tensor = root_matcher.outputs[0]
                matched_tensors[root_tensor] = node.outputs[0]
            
            matched_nodes_list.append(matched_nodes)
            matched_tensors_list.append(matched_tensors)

        return matched_nodes_list, matched_tensors_list


class MatcherBuilder:
    @staticmethod
    def build(
        self,
        matcher_root: TensorMatcher | NodeMatcher,
        name: str
    ) -> GraphMatcher:
        if isinstance(matcher_root, TensorMatcher):
            if matcher_root.producer is None:
                raise ValueError("root TensorMatcher must have a producer")
            matcher_root = matcher_root.producer
        
        vis: set[NodeMatcher] = {matcher_root}
        que: list[NodeMatcher] = list(vis)
        while que:
            cur = que.pop(0)
            for matcher_node_in in cur.inputs:
                if matcher_node_in.producer is None:
                    continue
                if matcher_node_in.producer not in vis:
                    vis.add(matcher_node_in.producer)
                    que.append(matcher_node_in.producer)
            if cur.outputs:
                for matcher_tensor_out in cur.outputs:
                    for consumer in matcher_tensor_out.consumers:
                        if consumer not in vis:
                            vis.add(consumer)
                            que.append(consumer)
        
        in_degree = {}
        out_degree = {}
        tensor_map = {}

        for n in list(vis):
            for t in n.inputs:
                in_degree[t.name] += 1
                tensor_map.setdefault(t.name, []).append(n)
            for t in n.outputs:
                out_degree[t.name] += 1
                tensor_map.setdefault(t.name, []).append(n)
        
        inputs = [
            tensor_map[name]
            for name in in_degree
            if out_degree[name] == 0
        ]
        outputs = [
            tensor_map[name]
            for name in out_degree
            if in_degree[name] == 0
        ]

        return GraphMatcher(list(vis), inputs, outputs, name)