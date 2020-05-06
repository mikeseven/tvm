import gzip
import os

import numpy as np
import tvm
from sima.relay_viz import relay_viz

from tvm import relay


def test_conv_bias():
    """ Testing quantization of Conv2D + BiasAdd at Relay level """

    host = "llvm -mcpu=core-avx2"
    target = host
    ctx = tvm.context(target)

    model_name='tvm_model'
    temp_dir = os.path.join(os.path.expanduser('~'), model_name)

    try:
        os.mkdir(temp_dir)
    except:
        pass

    data = relay.var("data", shape=(1, 16, 64, 64))  # NCHW
    conv = relay.nn.conv2d(data, relay.var("weight"),
                           kernel_size=(3, 3),
                           padding=(1, 1),
                           channels=16)

    bias = relay.var("bias", shape=(1, 16, 1, 1))  # vector of NxOx1x1 (must be 4D here, no broadcasting)
    biasAdd = relay.add(conv, bias)
    out = relay.nn.relu(biasAdd)

    # dummy data for all free vars
    data_val = np.random.uniform(size=(1, 16, 64, 64)).astype('float32')
    weight_val = np.random.uniform(size=(16, 16, 3, 3), low=-10, high=10).astype('float32')
    bias_val = np.random.uniform(size=(1, 16, 1, 1), low=-1, high=1).astype('float32')

    # wrap into a function and make a module
    f = relay.Function(relay.analysis.free_vars(out), out)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)
    params = {'weight': weight_val,  # dict of constants in the graph
              'bias': bias_val}
    print("Original:\n" + str(mod['main']))
    # print(params)
    relay_viz(mod, os.path.join(temp_dir, f"{model_name}_orig"), format='svg')

    # compile
    graph, lib, params = relay.build(mod, target, params=params)

    ## evaluate original
    interp = relay.create_executor('vm', mod, ctx, target)  # ('debug')
    res = interp.evaluate(mod['main'])(data_val, **params).asnumpy()  # evaluate f(data) and bind params

    ## quantize
    # weights are quantized symmetric in restricted range [-127,127]
    with relay.quantize.qconfig(skip_conv_layers=[],
                                nbit_input=8,
                                nbit_weight=8,
                                nbit_activation=32,
                                dtype_input="int8",
                                dtype_weight="int8",
                                dtype_activation="int32",
                                calibrate_mode='global_scale',
                                global_scale=8.0,
                                rounding='TONEAREST'):
        qmod = relay.quantize.quantize(mod, params)
    print("Quantized:\n" + str(qmod))
    relay_viz(qmod, os.path.join(temp_dir, f"{model_name}_quant"), format='svg')

    # compile
    qgraph, qlib, qparams = relay.build(qmod, target, params=params)
    _v = qparams['p0'].asnumpy()
    print(f"Quantized weights: {_v.shape} {_v.dtype}: min={np.amin(_v)} max={np.amax(_v)}")
    _v = qparams['p1'].asnumpy()
    print(f"Quantized bias: {_v.shape} {_v.dtype}: min={np.amin(_v)} max={np.amax(_v)}")

    ## evaluate quantized
    qres = interp.evaluate(qmod['main'])(data_val).asnumpy()

    ## validate results
    tvm.testing.assert_allclose(res, qres)

    ## now write to SiMa
    # TODO map ops to qnn ops
    # TODO validate execution of qnn graph
    # TODO write qnn graph as SiMa IR

    # group nodes
    qsima = sima_patterns(qmod['main'])
    print("Combined:\n" + str(qsima))
    relay_viz(qsima, os.path.join(temp_dir, f"{model_name}_sima"), format='svg')



def sima_patterns(mod):
    """Extract compute graph patterns suitable for SiMa nodes."""
    def run_opt_pass(expr, opt_pass):
        assert isinstance(opt_pass, tvm.transform.Pass)
        mod = tvm.IRModule.from_expr(expr)
        mod = opt_pass(mod)
        entry = mod["main"]
        return entry if isinstance(expr, relay.Function) else entry.body

    def _make_requantize_pattern(x=relay.var('x'), zp=relay.var('zero_point'), sh=relay.var('right_shift')):
        """Create a pattern to match the following graph.

           add(zero-point:int32)
             |
          right_shift(shift:int32)
             |
          clip(-127,127)
             |
          cast to int8
        """
        add_node = relay.add(x, zp)
        right_shift_node = relay.right_shift(add_node, sh)
        clip_node = relay.clip(right_shift_node, a_min=np.iinfo(np.int8).min, a_max=np.iinfo(np.int8).max)
        cast_node = relay.cast(clip_node, dtype='int8')
        return cast_node

    def _make_conv_bias_relu_pattern(x=relay.var('x'), weight=relay.var('weight'), bias=relay.var('bias')):
        """Create a pattern to match the following graph.

           conv2d
             |
            add
             |
           relu
        """
        conv_node = relay.nn.conv2d(x, weight)
        add_node = relay.add(conv_node, bias)
        r = relay.nn.relu(add_node)
        return r

    def _make_conv_requantize_pattern():
        """Create a pattern to match the following graph.

           conv2d
             |
          requantize
        """
        x = relay.var('x')
        weight = relay.var('weight')
        bias = relay.var('bias')
        zp = relay.var('zero_point')
        sh = relay.var('right_shift')
        cbir = _make_conv_bias_relu_pattern(x, weight, bias)
        req = _make_requantize_pattern(cbir, zp, sh)
        return req

    def _make_quantize_input():
        """Create a pattern to match the following graph.

           multiply - round - clip - cast
        """
        x = relay.var('x')
        const = relay.var('const')
        mul_node = relay.multiply(x, const)
        rnd_node = relay.round(mul_node)
        clip_node = relay.clip(rnd_node, a_min=np.iinfo(np.int8).min, a_max=np.iinfo(np.int8).max)
        cast_node = relay.cast(clip_node, dtype='int8')
        return cast_node

    def _make_dequantize_output():
        """Create a pattern to match the following graph.

           cast - multiply
        """
        x = relay.var('x')
        cast_node = relay.cast(x, dtype='float32')
        const = relay.var('const')
        mul_node = relay.multiply(cast_node, const)
        return mul_node

    pattern_table = [
        # ("requantize", _make_requantize_pattern()),
        # ("conv_bias_relu", _make_conv_bias_relu_pattern())
        ("sima.conv2d", _make_conv_requantize_pattern()),
        ("quantize_input", _make_quantize_input()),
        ("dequantize_output", _make_dequantize_output()),
    ]

    result = run_opt_pass(mod, relay.transform.MergeComposite(pattern_table))
    # result = run_opt_pass(result, relay.transform.InferType())

    return tvm.IRModule.from_expr(result)


######
#
# Misc utilities
#
######
def save_module(model_name, path, mod, graph=None, params=None, lib=None):
    with open(os.path.join(path, f'{model_name}_ir.txt'), 'w') as outfile:
        outfile.write(str(mod['main']))

    if graph:
        with open(os.path.join(path, f"{model_name}_graph.json"), "w") as fo:
            fo.write(graph)

    if params:
        with gzip.GzipFile(mode='wb', filename=os.path.join(path, f"{model_name}_params.gz")) as fo:
            fo.write(relay.save_param_dict(params))

    if lib:
        # lib.save(os.path.join(path, f"{model_name}_bin.o"))
        lib.export_library(os.path.join(path, f"{model_name}_bin.so"))


def load_module(model_name, path, mod, params, lib=None):
    loaded_json = open(os.path.join(path, f"{model_name}_deploy_graph.json")).read()
    loaded_lib = tvm.runtime.load_module(os.path.join(path, f"{model_name}_deploy_lib.so"))

    with gzip.GzipFile(mode='rb', filename=os.path.join(path, f"{model_name}_deploy_param.params.gz")) as fo:
        loaded_params = bytearray(fo.read())

    return loaded_json, loaded_lib, loaded_params


if __name__ == "__main__":
    test_conv_bias()
