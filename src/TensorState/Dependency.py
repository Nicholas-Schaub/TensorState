# This code was heavily inspired by torch-pruning
# https://github.com/VainF/Torch-Pruning/blob/master/torch_pruning/dependency.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torchvision
from grandalf.graphs import Edge, Graph, Vertex, graph_core
from pydantic import BaseModel

module_io = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


class NodeError(Exception):
    pass


class GradientData(BaseModel):
    name: str
    grad_fn: torch.autograd.graph.Node

    class Config:
        arbitrary_types_allowed = True


class ModuleData(GradientData):
    module: torch.nn.Module


class ElementNode(Vertex):
    _module_type: Tuple[Union[Type[torch.nn.Module], str], ...] = tuple()
    _data_type: Tuple[Type, ...] = (ModuleData, GradientData)
    traverse_node: bool = True
    data: Union[GradientData, ModuleData]
    TYPES: Dict[Union[Type[torch.nn.Module], str], OpNode] = {}

    def __init__(self, data: Union[GradientData, ModuleData]):
        assert isinstance(data, self._data_type)
        if len(self._module_type) > 0:
            success = False
            for mtype in self._module_type:
                if isinstance(mtype, str) and mtype in data.grad_fn.name().lower():
                    success = True
                    break

                elif isinstance(data, ModuleData) and isinstance(data.module, mtype):
                    success = True
                    break
            if not success:
                raise AssertionError

        super().__init__(data)

    def __str__(self) -> str:
        return str(f"class={self.__class__.__name__}, {self.data}")

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def register(
        cls,
        module_type: Union[
            Type[torch.nn.Module], str, Tuple[Union[Type[torch.nn.Module], str], ...]
        ],
        func: Optional[ElementNode] = None,
    ):
        if func is None:
            return lambda x: cls.register(module_type=module_type, func=x)

        assert module_type not in cls.TYPES, f"Module {module_type} already registered."

        if not isinstance(module_type, tuple):
            module_type = (module_type,)

        func._module_type = tuple()
        for mt in module_type:
            cls.TYPES[mt] = func
            func._module_type = func._module_type + (mt,)

    @classmethod
    def _default_new(cls, data):
        return ElementNode(data)

    @classmethod
    def new(cls, data: Union[GradientData, ModuleData]):
        print(data)
        cls_types = tuple(t for t in cls.TYPES if not isinstance(t, str))
        str_types = tuple(t for t in cls.TYPES if isinstance(t, str))
        if isinstance(data, ModuleData) and isinstance(data.module, tuple(cls_types)):
            for t, c in cls.TYPES.items():
                if isinstance(t, str):
                    continue
                try:
                    if isinstance(data.module, t):
                        node = c(data)
                        print(node)
                        return node
                except NodeError:
                    pass
        elif any(s in data.grad_fn.name().lower() for s in str_types):
            for s, c in cls.TYPES.items():
                if not isinstance(s, str):
                    continue
                try:
                    if s in data.grad_fn.name().lower():
                        node = c(data)
                        print(node)
                        return node
                except NodeError:
                    pass

        return cls._default_new(data)

    def _upstream_dendrites(self, vertex: Union[ElementNode, OpNode]):
        for edge in vertex.e_in():
            vertex = edge.v[0]
            if vertex.traverse_node:
                # if vertex == self:
                #     continue
                dendrites = vertex._upstream_dendrites(vertex)
                if dendrites is not None:
                    return dendrites
            else:
                return vertex.neurons()

        return 0

    def _downstream_neurons(self, vertex: Union[ElementNode, OpNode]):
        for edge in vertex.e_out():
            vertex = edge.v[1]
            if vertex.traverse_node:
                neurons = vertex._downstream_neurons(vertex)
                if neurons is not None:
                    return neurons
            else:
                return vertex.dendrites()

        return 0

    def dendrites(self):
        try:
            dendrites = self._upstream_dendrites(self)
        except RecursionError:
            print(self)
            raise
        return dendrites

    def neurons(self):
        try:
            neurons = self._downstream_neurons(self)
        except RecursionError:
            print(self)
            raise
        return neurons

    def apoptosis(self, idxs: List[int]):
        return None, idxs

    def prune(self, idxs: List[int]):
        return None, idxs

    def linked_neurons(
        self,
        idxs: Union[Tuple[Tuple[int], ...], Tuple[Tuple[Tuple, ...], ...], None] = None,
    ) -> Union[Tuple[Tuple[int], ...], Tuple[Tuple[Tuple, ...], ...]]:
        if idxs is None:
            return tuple((i,) for i in range(self.neurons()))
        else:
            if len(idxs) != self.neurons():
                raise ValueError(
                    f"""Node: {self}
                    Expected idxs to be size {self.neurons()} but was size {len(idxs)}"""
                )
            return idxs

    def nd_index(self, idxs: List[int], neurons: Optional[int] = None):
        if neurons is None:
            neurons = self.neurons()

        scale = self.dendrites() / neurons

        if scale == 1:
            return idxs
        elif scale > 1:
            out_idxs = []
            for idx in idxs:
                out_idxs.extend([idx + scale * i for i in range(int(scale))])
        else:
            dendrites = self.dendrites()
            out_idxs = [idx for idx in idxs if idx < dendrites]

            # Verify dendrites indices are linked
            assert (
                len(out_idxs) / len(idxs) == scale
            ), f"{self.__class__.__name__}: scale={scale}, dendrites={self.dendrites()}, neurons={neurons}"
            for idx in out_idxs:
                for i in range(int(scale)):
                    assert (idx + scale * i) in idxs

        return out_idxs


class OpNode(ElementNode):
    _module_type = (torch.nn.Module,)
    _data_type = (ModuleData,)
    traverse_node: bool = False
    TYPES: Dict[Union[Type[torch.nn.Module], str], OpNode] = {}

    @classmethod
    def _default_new(cls, data):
        return ElementNode.new(data)

    def dendrites(self):
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def neurons(self):
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def prune(self, idxs: List[int]):
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def nd_index(self, idxs: List[int], neurons: Optional[int] = None):
        return None


@ElementNode.register(torch.nn.modules.batchnorm._BatchNorm)
class BatchNormNode(ElementNode):
    def dendrites(self):
        return self.data.module.num_features

    neurons = dendrites

    def prune(self, idxs: List[int]):
        assert isinstance(self.data, ModuleData)
        assert isinstance(self.data.module, torch.nn.modules.batchnorm._BatchNorm)

        layer = self.data.module
        keep_idxs = list(set(range(layer.num_features)) - set(idxs))
        keep_idxs.sort()
        layer.num_features = layer.num_features - len(idxs)

        if layer.running_mean is not None:
            layer.running_mean = layer.running_mean.data.clone()[keep_idxs]

        if layer.running_var is not None:
            layer.running_var = layer.running_var.data.clone()[keep_idxs]

        if layer.affine:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])

        return keep_idxs, idxs


@ElementNode.register(
    (
        torch.nn.modules.pooling._AdaptiveMaxPoolNd,
        torch.nn.modules.pooling._AdaptiveAvgPoolNd,
    )
)
class AdaptivePoolNode(ElementNode):
    pass


@ElementNode.register(torchvision.ops.misc.Permute)
class PermuteNode(ElementNode):
    pass


@ElementNode.register(("reshape", "view"))
class ReshapeNode(ElementNode):
    traverse_node = False

    def linked_neurons(
        self,
        idxs: Union[Tuple[Tuple[int], ...], Tuple[Tuple[Tuple, ...], ...], None] = None,
    ):
        idxs = super().linked_neurons(idxs)
        scale = self.neurons() / self.dendrites()
        dendrites = self.dendrites()

        return tuple(tuple(idxs[i::dendrites]) for i in range(dendrites))


@OpNode.register(torch.nn.modules.conv._ConvNd)
class ConvNode(OpNode):
    def __init__(self, data: Union[GradientData, ModuleData]):
        super().__init__(data)
        assert isinstance(data, ModuleData)
        if data.module.groups != 1:
            raise NodeError

    def dendrites(self):
        return self.data.module.in_channels

    def neurons(self):
        return self.data.module.out_channels

    def apoptosis(self, idxs: List[int]):
        assert isinstance(self.data, ModuleData)
        assert isinstance(self.data.module, torch.nn.modules.conv._ConvNd)

        layer = self.data.module
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        layer.out_channels -= len(idxs)

        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])

        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])

        return keep_idxs, idxs

    def prune(self, idxs: List[int]):
        assert isinstance(self.data, ModuleData)
        assert isinstance(self.data.module, torch.nn.modules.conv._ConvNd)

        layer = self.data.module
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        keep_idxs.sort()
        layer.in_channels = layer.in_channels - len(idxs)

        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])

        return keep_idxs, idxs


@ElementNode.register(torch.nn.modules.conv._ConvNd)
class ConvGroupNode(ElementNode):
    def __init__(self, data: Union[GradientData, ModuleData]):
        super().__init__(data)
        assert isinstance(data, ModuleData)
        if (
            data.module.groups != data.module.in_channels
            and data.module.in_channels != data.module.out_channels
        ):
            raise NodeError

    def dendrites(self):
        return self.data.module.in_channels

    def neurons(self):
        return self.data.module.out_channels

    def apoptosis(self, idxs: List[int]):
        assert isinstance(self.data, ModuleData)
        assert isinstance(self.data.module, torch.nn.modules.conv._ConvNd)

        layer = self.data.module
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        layer.out_channels -= len(idxs)

        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])

        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])

        layer.in_channels = layer.out_channels
        layer.groups = layer.out_channels

        return keep_idxs, idxs


@OpNode.register(torch.nn.Linear)
class LinearNode(OpNode):
    def apoptosis(self, idxs: List[int]):
        assert isinstance(self.data, ModuleData)
        assert isinstance(self.data.module, torch.nn.Linear)

        layer = self.data.module
        keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        keep_idxs.sort()
        layer.out_features -= len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])

        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])

        return keep_idxs, idxs

    def dendrites(self):
        return self.data.module.in_features

    def neurons(self):
        return self.data.module.out_features

    def prune(self, idxs: List[int]):
        assert isinstance(self.data, ModuleData)
        assert isinstance(self.data.module, torch.nn.Linear)

        layer = self.data.module
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        keep_idxs.sort()
        layer.in_features = layer.in_features - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])

        return keep_idxs, idxs


class Dependency(Edge):
    x: Union[OpNode, ElementNode]
    y: Union[OpNode, ElementNode]
    w: Union[int, float] = 1
    data: Optional[Any] = None
    connect: bool = False

    def __init__(
        self,
        x: Union[OpNode, ElementNode],
        y: Union[OpNode, ElementNode],
        w: Union[int, float] = 1,
        data: Optional[Any] = None,
        connect: bool = False,
    ):
        assert isinstance(x, (OpNode, ElementNode))
        assert isinstance(y, (OpNode, ElementNode))
        super().__init__(x, y, w, data, connect)

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return str(self.data)


class GroupGraph(graph_core):
    def apoptosis(self, idxs: List[int]):
        stack_type = Tuple[Union[OpNode, ElementNode], List[int]]

        # Get terminal nodes
        leaves = self.leaves()
        roots = self.roots()
        process_stack: List[stack_type] = [(leaf, idxs) for leaf in leaves]
        visited: Set[Union[OpNode, ElementNode]] = set()

        apoptosis_stack: List[stack_type] = []

        # Get apoptosis information
        while len(process_stack) > 0:
            node, indices = process_stack.pop()

            if node in visited:
                print(f"Node in visited")
                continue
            else:
                visited.add(node)

            print(f"Processing Node: {node}")
            new_indices = node.nd_index(indices)

            for edge in node.e_in():
                vertex: Union[OpNode, ElementNode] = edge.v[0]
                process_stack.append(
                    (vertex, new_indices if new_indices is not None else indices)
                )

            if isinstance(node.data, ModuleData):
                apoptosis_stack.append((node, indices))

        # Apoptosis and prune on designated nodes
        visited = set()
        for node, indices in apoptosis_stack:
            if node in visited:
                continue
            else:
                visited.add(node)

            print(f"pruning node: {node}, {indices}")

            if node not in leaves:
                node.apoptosis(indices)

            if node not in roots:
                node.prune(indices)

    def linked_neurons(self):
        leaves = self.leaves()
        roots = self.roots()

        if len(list(self.V())) == 1 or leaves[0].dendrites() == roots[0].neurons():
            return [tuple([(i,) for i in range(leaves[0].neurons())])]

        if len(leaves) != 1:
            raise NotImplementedError("Multiple output nodes are not yet supported")

        process_stack = []
        for node in leaves:
            indices = list([(i,) for i in range(node.dendrites())])
            for edge in node.e_in():
                process_stack.append((edge.v[0], indices))

        visited = set()
        linkages = []
        while len(process_stack) > 0:
            node, indices = process_stack.pop()

            if node in visited:
                continue
            else:
                visited.add(node)

            new_indices = node.linked_neurons(indices)

            for edge in node.e_in():
                vertex = edge.v[0]
                if vertex in roots:
                    linkages.append(new_indices)
                else:
                    process_stack.append((vertex, new_indices))

        return linkages


class ModuleGraph(Graph):
    model: torch.nn.Module
    _visit_count: Dict[torch.nn.Module, int] = {}
    _grad_trace: Dict[Any, torch.nn.Module] = {}
    _block_hook: bool = True

    def __init__(
        self,
        model: torch.nn.Module,
        inp_tensor: module_io = torch.ones((1, 3, 256, 256)),
    ):
        self.model = model

        # Find module associated gradients
        self._visit_count = {module: 0 for module in model.modules()}

        hooks = [
            module.register_forward_hook(self)
            for module in model.modules()
            if not list(module.children())
        ]

        self._block_hook = False

        self.model.eval()
        device = next(model.parameters()).device
        out: torch.Tensor = self.model(inp_tensor.to(device))

        self._block_hook = True

        for hook in hooks:
            hook.remove()

        # Trace the network and generate the nodes and edges
        gradients: List[torch.autograd.graph.Node] = [out.grad_fn]
        visited_nodes: List[torch.autograd.graph.Node] = []

        if out.grad_fn in self._grad_trace:
            node = OpNode.new(
                data=ModuleData(
                    module=self._grad_trace[out.grad_fn],
                    grad_fn=out.grad_fn,
                    name=out.grad_fn.name(),
                )
            )
        else:
            node = OpNode.new(
                data=GradientData(grad_fn=out.grad_fn, name=out.grad_fn.name())
            )

        nodes: Dict[Any, Union[OpNode, ElementNode]] = {out.grad_fn: node}
        edges: List[Dependency] = []

        while len(gradients) > 0:
            grad_fn = gradients.pop()
            if grad_fn in visited_nodes:
                continue
            else:
                visited_nodes.append(grad_fn)

            node = nodes[grad_fn]

            for gf in grad_fn.next_functions:
                gf = gf[0]
                if gf is None:
                    continue
                if gf in nodes:
                    upstream_node = nodes[gf]
                else:
                    if gf in self._grad_trace:
                        upstream_node = OpNode.new(
                            data=ModuleData(
                                module=self._grad_trace[gf],
                                grad_fn=gf,
                                name=gf.name(),
                            )
                        )
                    else:
                        if "accumulategrad" in gf.name().lower():
                            continue
                        upstream_node = OpNode.new(
                            data=GradientData(grad_fn=gf, name=gf.name())
                        )

                    nodes[gf] = upstream_node

                # if upstream_node.data.grad_fn != node.data.grad_fn:
                edges.append(Dependency(x=upstream_node, y=node))

                gradients.append(gf)

        super().__init__(V=list(nodes.values()), E=edges)

        groups = Graph(V=list(nodes.values()), E=edges)

        vertices = list(groups.V())

        # Create disjoint sets along OpNodes
        for vertex in vertices:
            if isinstance(vertex, OpNode):
                temp_node = OpNode.new(data=vertex.data)
                groups.add_vertex(temp_node)
                for edge in vertex.e_out():
                    temp_edge = Dependency(x=temp_node, y=edge.v[1])
                    groups.add_edge(temp_edge)
                    groups.remove_edge(edge)

        self.groups = []
        for component in groups.C:
            # Cleanup nodes, remove duplicates and merge connections
            modules = [
                m.data.grad_fn for m in component.V() if isinstance(m.data, ModuleData)
            ]
            duplicates = [m for m in modules if modules.count(m) > 1]
            while len(duplicates) > 0:
                grad_fn = duplicates.pop()

                vertices: List[Vertex] = [
                    v
                    for v in component.V()
                    if isinstance(v.data, ModuleData) and v.data.grad_fn == grad_fn
                ]
                if len(vertices) == 0:
                    continue
                vertex = vertices.pop()
                dependencies = []
                for v in vertices:
                    for e in component.E():
                        if e.v[1] == v:
                            dependencies.append(Dependency(x=e.v[0], y=vertex))
                        elif e.v[0] == v:
                            dependencies.append(Dependency(x=vertex, y=e.v[1]))

                    for dep in dependencies:
                        component.add_edge(dep)

                    component.remove_vertex(v)

                duplicates = [d for d in duplicates if d != grad_fn]

            self.groups.append(GroupGraph(list(component.V()), list(component.E())))

    def __call__(
        self, module: torch.nn.Module, inputs: module_io, outputs: module_io
    ) -> None:
        if self._block_hook:
            raise RuntimeError(
                "Module is intended to be used as a forward hook and "
                + "should not be directly called."
            )
        self._visit_count[module] += 1
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        self._grad_trace[outputs.grad_fn] = module
