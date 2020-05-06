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
from sima.utils import get_model_tf, simplify_graph, save_module
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, default=None, required=True,
                        help='TensorFlow .pb model file')
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
    mod, params, _ = get_model_tf(filename, layout="NHWC")  # TF models are always NHWC
    if not mod:
        print(f"ERROR: Could not load model in {filename}.")
        sys.exit(-1)

    t.update()
    t.set_postfix_str(f"Converting to TVM IR.")

    host = "llvm -mcpu=core-avx2"
    target = host

    ctx = tvm.context(target)

    # convert all ops' layout to NCHW for Sima compiler
    func = run_opt_pass(mod['main'], relay.transform.ConvertLayout('NCHW'))
    mod = tvm.IRModule.from_expr(func)

    mod = simplify_graph(mod, params)
    # print(mod)

    t.update()
    t.set_postfix_str(f"Extract SiMa patterns.")
    mod = sima_patterns(mod['main'])

    # with relay.build_config(opt_level=3):
    #     graph, lib, params = relay.build(mod, target, host, params=params)

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
