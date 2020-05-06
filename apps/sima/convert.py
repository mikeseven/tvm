import argparse
import gzip
import os
import sys
import logging

import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime

from sima.relay_viz import relay_viz
from sima.utils import get_model_tf, simplify_graph
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("compile_engine").setLevel(logging.CRITICAL)
logging.getLogger("strategy").setLevel(logging.CRITICAL)
logging.getLogger("topi").setLevel(logging.CRITICAL)
logging.getLogger("autotvm").setLevel(logging.CRITICAL)
logging.getLogger("relay").setLevel(logging.CRITICAL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, default=None, required=True,
                        help='TensorFlow .pb model file')
    # parser.add_argument("--in_name", type=str, default=None, required=True,
    #                     help="Input op name")
    # parser.add_argument("--in_shape", type=int, default=None, nargs='+', required=True,
    #                     help="Input op shape e.g. 1 1024 2048 3 for TF NHWC layout")
    # parser.add_argument("--out_name", type=str, default=None, required=True,
    #                     help="Output node name")
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
    mod, params, _ = get_model_tf(filename)  #, args.in_shape, in_node=args.in_name, out_node=args.out_name)

    if not mod:
        print(f"ERROR: Could not load model in {filename}.")
        sys.exit(-1)

    t.update()
    t.set_postfix_str(f"Converting to TVM IR.")

    host = "llvm -mcpu=core-avx2"
    target = host

    ctx = tvm.context(target)
    mod = simplify_graph(mod, params, target)

    with relay.build_config(opt_level=3):
        graph, lib, compiled_params = relay.build(mod, target, host, params=params)

    t.update()
    t.set_postfix_str(f"Saving IR and params.")
    tvm_out_basename = args.out if args.out else os.path.join(model_path, model_name)
    with open(f"{tvm_out_basename}_ir.txt", 'w') as f:
        f.write(str(mod['main']))

    with gzip.GzipFile(mode='wb', filename=f"{tvm_out_basename}_params.gz") as f:
        f.write(relay.save_param_dict(compiled_params))

    if args.viz:
        relay_viz(mod, os.path.join(model_path, f"{model_name}"), format='svg')

    t.update()
    t.set_postfix_str("Done.")

