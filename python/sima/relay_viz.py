from graphviz import Digraph
from tvm import relay


def relay_viz(mod, graph_name, directory='.', format='svg'):
    def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        if isinstance(node, relay.op.op.Op):
            return
        node_dict[node] = len(node_dict)

    dot = Digraph(format=format)
    dot.attr(rankdir='BT')  # directed graph arranged from bottom to top (BT)
    dot.attr('node', shape='box')  # default style

    node_dict = {}
    relay.analysis.post_order_visit(mod if callable(mod) else mod['main'], lambda node: _traverse_expr(node, node_dict))
    for node, node_idx in node_dict.items():
        if isinstance(node, relay.expr.Var):
            # print(
            #     f'node_idx: {node_idx}, Var(name={node.name_hint}, type=Tensor[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}])')
            dot.node(str(node_idx),
                     f'{node.name_hint}:\nVar[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]',
                     shape='underline')
        elif isinstance(node, relay.expr.Constant):
            # print(
            #     f'node_idx: {node_idx}, Const(type=Tensor[{tuple(node.data.shape)}, {node.data.dtype}])')
            if node.data.shape:
                dot.node(str(node_idx),
                         f'Const[{tuple(node.data.shape)}, {node.data.dtype}]',
                         shape='underline', style='filled')
            else:
                dot.node(str(node_idx),
                         f'Const[{node.data.dtype}] = {node.data.asnumpy()}',
                         shape='underline', style='filled')
        elif isinstance(node, relay.expr.Call):
            args = [node_dict[arg] for arg in node.args]
            if isinstance(node.op, relay.Function):
                # print(f'node_idx: {node_idx}, Call(Function({node_dict[node.op.body]}))')
                if node.op.attrs:
                    dot.node(str(node_idx), f'Call(Func {node_dict[node.op.body]} - {node.op.attrs.items()})',
                             shape='egg', style='filled', color='red', fontcolor='white')
                else:
                    dot.node(str(node_idx), f'Call(Func {node_dict[node.op.body]})',
                             shape='egg', style='filled', color='red', fontcolor='white')

            else:
                # print(f'node_idx: {node_idx}, Call(op_name={node.op.name}, args={args})')
                dot.node(str(node_idx), f'{node.op.name}', shape='oval', style='filled',
                         color='lightblue2' if node.op.name.startswith("nn.") else "lightblue3")
            for arg in args:
                dot.edge(str(arg), str(node_idx))
        elif isinstance(node, relay.Function):  # function declaration
            # print(f'node_idx: {node_idx}, Function(body={node_dict[node.body]})')
            if node.attrs:
                dot.node(str(node_idx), f'Func {node_dict[node.body]} - {node.attrs.items()}',
                         shape='egg', style='filled', color='coral')
            else:
                dot.node(str(node_idx), f'Func {node_dict[node.body]}',
                         shape='egg', style='filled', color='coral')
            dot.edge(str(node_dict[node.body]), str(node_idx))
        elif isinstance(node, relay.expr.TupleGetItem):
            # print(f'node_idx: {node_idx}, TupleGetItem(tuple={node_dict[node.tuple_value]}, idx={node.index})')
            dot.node(str(node_idx), f'TupleGetItem(idx={node.index})',
                     shape='box', style='filled', color='green')
            dot.edge(str(node_dict[node.tuple_value]), str(node_idx))
        elif isinstance(node, relay.expr.Tuple):
            fields = [node_dict[f] for f in node.fields]

            # print(f'node_idx: {node_idx}, TupleGetItem(tuple={node_dict[node.tuple_value]}, idx={node.index})')
            dot.node(str(node_idx), 'Tuple', shape='doublecircle', style='filled', color='lightgreen')
            for f in fields:
                dot.edge(str(f), str(node_idx))
        else:
            raise RuntimeError(f'Unknown node type. node_idx: {node_idx}, node: {type(node)}')

    # print(dot.source)  # graphviz source
    dot.render(filename=graph_name + '.gv', directory=directory, format=format, quiet=True)
