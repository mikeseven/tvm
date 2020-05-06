import argparse
import gzip
import os
import sys
import logging

import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime
from tvm.relay.testing import run_opt_pass

from sima.relay_viz import relay_viz
from sima.utils import get_model_tf, save_module
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("compile_engine").setLevel(logging.CRITICAL)
logging.getLogger("strategy").setLevel(logging.CRITICAL)
logging.getLogger("topi").setLevel(logging.CRITICAL)
logging.getLogger("autotvm").setLevel(logging.CRITICAL)
logging.getLogger("relay").setLevel(logging.CRITICAL)


def sima_patterns(func):
    """Extract compute graph patterns suitable for SiMa nodes."""

    def _make_conv_bias_relu_pattern(data=relay.var('data'), weight=relay.var('weight'), bias=relay.var('bias')):
        """Create a pattern to match the following graph.

           conv2d -> add -> relu
        """
        f = relay.nn.conv2d(data, weight)
        f = relay.add(f, bias)
        f = relay.nn.relu(f)
        return f

    def _make_conv_transpose_bias_relu_pattern(data=relay.var('data'), weight=relay.var('weight'),
                                               bias=relay.var('bias')):
        """Create a pattern to match the following graph.

           transpose to HWC -> conv2d -> transpose to CHW -> add -> relu
        """
        f = relay.transpose(data, axes=[0, 3, 1, 2])
        f = relay.nn.conv2d(data, weight)
        f = relay.transpose(data, axes=[0, 2, 3, 1])
        f = relay.add(f, bias)
        f = relay.nn.relu(f)
        return f

    pattern_table = [
        ("conv_bias_relu", _make_conv_bias_relu_pattern()),
        ("hwc_conv_bias_relu", _make_conv_transpose_bias_relu_pattern()),
    ]

    result = run_opt_pass(func, relay.transform.MergeComposite(pattern_table))

    return tvm.IRModule.from_expr(result)


def simplify_graph(mod, params=None):
    """Simplify module for inference on MLA."""
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout('NCHW'),
                                    relay.transform.InferType(),  # these next 4 remove BN
                                    relay.transform.SimplifyInference(),
                                    relay.transform.FoldConstant(),
                                    relay.transform.FoldScaleAxis(),
                                    relay.transform.EliminateCommonSubexpr(),
                                    # relay.transform.FuseOps(fuse_opt_level=2)
                                    ])

    def _bind_params(func, params):
        """Bind the params to the expression."""
        name_dict = {}
        for arg in func.params:
            name = arg.name_hint
            if name in name_dict:
                name_dict[name] = None
            else:
                name_dict[name] = arg
        bind_dict = {}
        for k, v in params.items():
            if k not in name_dict:
                continue
            arg = name_dict[k]
            if arg is None:
                raise ValueError("Multiple args in the function have name %s" % k)
            bind_dict[arg] = relay.expr.const(v)
        return relay.expr.bind(func, bind_dict)

    if params:
        mod['main'] = _bind_params(mod['main'], params)

    with relay.build_config(opt_level=3):  # to make sure EliminateCommonSubexpr is executed (level 2)
        mod = seq(mod)
        # mod, params = relay.optimize(mod, target, params)

    return mod


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, default=None, required=True,
                        help='TensorFlow .pb model file')
    parser.add_argument("--in_name", type=str, default=None,
                        help="Input op name")
    parser.add_argument("--in_shape", type=int, default=None, nargs='+',
                        help="Input op shape e.g. 1 1024 2048 3 for TF NHWC layout")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Output node name")
    parser.add_argument("--out", type=str, default=None,
                        help="Output file name")
    parser.add_argument("--viz", action='store_true',
                        help="Output visualization of the graph as SVG file")
    args = parser.parse_args()

    mod = None
    params = None
    model_name = None
    model_path = None

    filename = args.tf
    if filename.startswith('~'):
        filename = os.path.expanduser(filename)
    model_path = os.path.dirname(filename)
    model_name, _ = os.path.splitext(os.path.basename(filename))

    if not os.path.isfile(filename):
        print(f"File {filename} does NOT exist")
        sys.exit(-1)

    t = tqdm(total=3, unit="step", bar_format="{l_bar}{bar}|{rate_fmt}{postfix}")
    t.set_postfix_str(f"Loading {filename}")
    mod, params, _ = get_model_tf(filename,
                                  in_node=args.in_name,
                                  in_shape=args.in_shape,
                                  out_node=args.out_name,
                                  layout="NHWC")  # TF models are always NHWC
    if not mod:
        print(f"ERROR: Could not load model in {filename}.")
        sys.exit(-1)

    relay_viz(mod, "/Users/mikael/tmp_convert_orig", format='svg')

    t.update()
    t.set_postfix_str(f"Converting to TVM IR.")

    target = host = "llvm"
    ctx = tvm.cpu()

    mod = simplify_graph(mod, params)

    # make sure tensors are all in the right layout
    # TODO unfortunately, LRN layout is wrong and crashes the build???
    with relay.build_config(opt_level=3):
        _, _, params = relay.build(mod, target, host, params=params)

    t.update()
    t.set_postfix_str(f"Extract SiMa patterns.")
    mod = sima_patterns(mod['main'])

    filename = args.out if args.out else os.path.join(model_path, model_name)
    t.update()
    t.set_postfix_str(f"Saving IR and params to {filename}.*")
    if filename.startswith('~'):
        filename = os.path.expanduser(filename)

    try:
        os.mkdir(os.path.dirname(filename))
    except:
        pass

    save_module(filename, mod, params=params)

    if args.viz:
        relay_viz(mod, filename, format='svg')
