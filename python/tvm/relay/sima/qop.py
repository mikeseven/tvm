import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing import run_infer_type

import logging

"""
A QOp wraps a group of ops in a graph to collect stats and determine correct quantizer params.
A QGraph wraps the whole model ie a graph of QOps.

"""


class QOp:
    def __init__(self, func: relay.Function, B: int = 8, signed=True, sym=False, dtype='float32'):
        self.func = func
        self.B = B
        self.signed = signed
        self.sym = sym
        self.dtype = dtype

        # quantization params for activations (dynamic, from calibration)
        class Stats:
            """Quantizer for values."""

            def __init__(self, B=8, signed=True, sym=False):
                self.min = np.finfo(np.float32).max
                self.max = np.finfo(np.float32).min
                self.signed = signed
                self.sym = sym
                self.B = B
                self.s = 1
                self.z = 0

            def update(self, mini, maxi):
                self.min = mini if mini < self.min else self.min
                self.max = maxi if maxi > self.max else self.max

            def power2(self):
                """Realize quantization parameters"""
                pot = 1 << self.B
                delta = self.max - self.min
                self.s = delta / pot
                if self.s < 0.5:  # TODO S=0 is wrong, we need to nudge and compensate error. I push it to 1 for now
                    self.s = 1
                self.z = 0 if self.sym else int(np.floor(-self.min / self.s + 0.5))
                self.s = int(np.floor(self.s + 0.5))

            def __repr__(self):
                return f"{{min={self.min}, max={self.max}, s={self.s}, z={self.z}}}"

        self.qp = {
            'a': Stats(B=self.B, signed=True, sym=False),
            'c': Stats(B=self.B, signed=True, sym=False),
            'b': Stats(B=self.B, signed=True, sym=True),
            'w': Stats(B=self.B, signed=True, sym=True),
        }
        self.n = 0  # should never happen

        # runtime
        self.vm = None
        self.mod = None
        self.params = None
        self.qmod = None  # final quantized module

    def func(self):
        return self.func

    def _update(self, p: str, data: np.array):
        if data is None:
            return
        if not (p in self.qp.keys()):
            return

        self.qp[p].update(data.min(), data.max())

    def set_params(self, params):
        self.params = params
        self._update('w', params['weight'])
        self._update('b', params['bias'])

    def calibrate(self, data, ctx, target):
        if not self.vm:
            # self.func = relay.Function(relay.analysis.free_vars(self.func), self.func)
            self.mod = tvm.IRModule.from_expr(self.func)
            self.vm = relay.create_executor('vm', self.mod, ctx, target)

        # run it with input data
        self._update('a', data)
        res = self.vm.evaluate(self.mod['main'])(data, **self.params if self.params else None).asnumpy()
        self._update('c', res)
        return res

    def realize(self):
        # compute power2 parameters
        for v in self.qp.values():
            v.power2()

        # compute SiMa params
        self.n = -int(np.floor(np.log2(self.qp['a'].s * self.qp['w'].s / self.qp['c'].s) + 0.5))

    def qmodule(self) -> tvm.IRModule:
        """Create quantized module.

        The idea is to write a GlobalVar since it is a named function.
        Only the parameters to the function are useful for n2a compiler.
        The body of the function is a functional representation of quantized compute operations TVM can compile.

        TODO likewise for SimaInputQuantize, SimaOutputDequantize, SimaRequantize
        TODO not ideal for now. Just a quick way to avoid doing it in C++ (but we should ultimately)
        """
        if not (self.qmod is None):
            return self.qmod

        # creates function out of expr
        # if not isinstance(self.func, relay.Function):
        #     func = relay.Function(relay.analysis.free_vars(self.func), self.func)

        mod = relay.transform.InferType()(tvm.IRModule.from_expr(self.func))  # get checked_type
        func = mod['main']
        data_shp = func.params[0].checked_type.shape
        weight_shp = func.params[1].checked_type.shape
        bias_shp = func.params[2].checked_type.shape

        qa = relay.var("qa", relay.TensorType((), dtype="int8"))  # determined at runtime
        qw = relay.var("qw", relay.TensorType(weight_shp, dtype="int8"))
        qb = relay.var("qb", relay.TensorType(bias_shp, dtype="int8"))

        # constants
        za = relay.var("za", relay.TensorType(data_shp, dtype="int8"))
        zc = relay.var("zc", relay.TensorType((), dtype="int8"))  # determined at runtime
        n = relay.const(self.n)

        f = relay.subtract(relay.cast(qa, "int32"), relay.cast(za, "int32"))
        f = relay.nn.conv2d(f, relay.cast(qw, "int32"))  # TODO true for any linear op
        f = relay.add(f, relay.cast(qb, "int32"))  # TODO bias if any
        f = relay.right_shift(f, relay.cast(n, "int32"))
        f = relay.add(f, relay.cast(zc, "int32"))
        f = relay.cast(f, "int8")
        f = relay.clip(f, -127, 127)

        func = relay.Function(relay.analysis.free_vars(f), f)
        mod = tvm.IRModule()
        mod[relay.GlobalVar("SimaConv")] = func
        return mod

    def astext(self) -> str:
        return str(self.qmodule())

    def __repr__(self) -> str:
        return f"{self.qp.items()} n={self.n}"


class QGraph(QOp):
    """QGraph is just a QOp that has children QOps"""

    def __init__(self, func: relay.Function = None, B: int = 8, signed=True, sym=False, dtype='float32'):
        super().__init__(func, B, signed, sym, dtype)
        self.children = []

        # runtime
        self.vm = None
        self.mod = None
        self.params = None
        self.mod_dirty = True

    def analyze(self):
        """Parse a model in self.func and creates QOp for quantizable ops.
        Maintain the graph with relationship between ops and QOps.
        """
        if not self.func:
            return

        # map <relay.expr, id>
        node_dict = {}

        # note: traversal is in DFS order
        def _visit_func(node):
            if node in node_dict:
                return
            if isinstance(node, relay.op.op.Op):
                return
            node_dict[node] = len(node_dict)

        relay.analysis.post_order_visit(self.func, _visit_func)
        for node, node_idx in node_dict.items():
            # print(f"*** %{node_idx} =", node)
            if isinstance(node, relay.expr.Call):
                """Call an operator or function. """
                args = [node_dict[arg] for arg in node.args]    # arguments of function
                print(f"Calling {node_idx}={node.op.name} with args {args}")
                # if node.attrs:
                #     fields=node.attrs.keys()
                #     print(f"{node.op.name}.attrs= {fields}")
                items = list(node_dict.items())
                for arg in args:
                    print(f"   {items[arg]}")

    def calibrate(self, dataset, ctx, target):
        # run it with input data from calibration dataset
        for data in dataset:
            self._update('a', data)
            for child in self.children:
                res = child.calibrate(data, ctx, target)
                self._update('c', res)

        # TODO compute quantization params
        self.mod_dirty = True

    # TODO
    def qmodule(self):
        if self.mod and not self.mod_dirty:
            return self.mod

        # data_val = np.random.uniform(size=(1, 16, 64, 64), low=-1, high=1).astype("float32")
        # weight_val = np.random.uniform(size=(16, 16, 3, 3), low=-10, high=10).astype('float32')
        # bias_val = np.random.uniform(size=(1, 16, 1, 1), low=-1, high=1).astype('float32')
        #
        # # params = (data_val, weight_val, bias_val)
        # params = {relay.const('data', data_val), relay.const('weight', weight_val), relay.const("bias", bias_val)}
        # # fn = params
        # for child in self.children:
        #     func = child.func()
        #     print("Original func:", func)
        #     func = relay.Function(relay.analysis.free_vars(func), func)
        #     print("Function func:", func)
        #     fn = relay.Call(func, params)
        #     print("Func(data):", fn)
        #     # fn = run_infer_type(fn)
        #
        # fn = run_infer_type(fn)
        # model = relay.Function(relay.analysis.free_vars(fn), fn)
        # self.mod = tvm.IRModule.from_expr(model)

        x = relay.var("x", relay.TensorType((), dtype="float32"))

        mod = tvm.IRModule()
        qinput = relay.GlobalVar("SimaQuantInput")
        mod[qinput] = self._quantize_input(2, 0)

        x = relay.var("x", relay.TensorType((), dtype="float32"))
        f = qinput(x)

        for child in self.children:
            child_mod = child.qmodule()
            # mod.update(child_mod)

            for gv in child_mod.get_global_vars():
                fn = child_mod[gv]
                mod[gv] = fn

        qoutput = relay.GlobalVar("SimaDequantOutput")
        mod[qoutput] = self._dequantize_output(5, 10)
        f = qoutput(f)

        mod["main"] = relay.Function(relay.analysis.free_vars(f), f)

        return mod

    def _quantize_input(self, input_scale, input_zero_point) -> relay.Function:
        """Creates a function that quantizes fp32 input to Sima quantized space.
        """
        r = relay.var("r", relay.TensorType((), dtype="float32"))
        sa = relay.const(input_scale, "float32")
        za = relay.const(input_zero_point, "float32")

        # q=r/sa + za
        f = relay.divide(r, sa)
        f = relay.add(f, za)
        f = relay.cast(f, "int8")
        f = relay.clip(f, -127, 127)

        f = relay.Function(relay.analysis.free_vars(f), f)
        return f

    def _dequantize_output(self, output_scale, output_zero_point) -> relay.Function:
        """Creates a function that dequantizes output from Sima quantized space to fp32 real space
        """
        q = relay.var("q", relay.TensorType((), dtype="int8"))
        sc = relay.const(output_scale, "float32")
        zc = relay.const(output_zero_point, "float32")

        # r = sc (q-zc)
        f = relay.subtract(relay.cast(q, "float32"), zc)
        f = relay.multiply(f, sc)

        f = relay.Function(relay.analysis.free_vars(f), f)
        return f

    def astext(self) -> str:
        return str(self.qmodule())
