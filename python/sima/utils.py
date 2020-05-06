import os
import gzip
import numpy as np
import logging

# TVM
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime as runtime
from tvm.relay.testing import run_infer_type

# Tensorflow
import tensorflow as tf

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf


#
# Misc. utilities
#
def normalize_image(image, mean, std, swap_channel=False, dtype='float32'):
    """Take an int8 image in HWC (eg PIL) and normalize it.
     Returns NHWC or NCHW data as dtype (default: float32) by adding N axis.
     Optionally, swap channel axis e.g. HWC to CHW
    """
    image = np.array(image) - np.array(mean)
    image /= np.array(std)
    if swap_channel:
        image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :].astype(dtype)
    return image


def normalize_image_symmetric(x):
    """ int8 NHWC image converted to in [-1,1] """
    x[:, :, :, 0] = 2.0 / 255.0 * x[:, :, :, 0] - 1
    x[:, :, :, 1] = 2.0 / 255.0 * x[:, :, :, 1] - 1
    x[:, :, :, 2] = 2.0 / 255.0 * x[:, :, :, 2] - 1
    return x


def tf_infer(tf_graph, input_data, input_node, output_node):
    """Runs TensorFlow inference """
    logging.info(f"Executing TF model on CPU")
    with tf_compat_v1.Session(graph=tf_graph) as sess:
        x = sess.graph.get_tensor_by_name(f'{input_node}:0')  # input node
        y = sess.graph.get_tensor_by_name(f'{output_node}:0')  # output node

        # run graph for predictions
        sess.run(tf_compat_v1.global_variables_initializer())  # initialize variables
        tf_output = sess.run(y,  # output tensor
                             feed_dict={x: input_data}  # inputs
                             )
    return tf_output


#
# Loading models from TF, Pytorch, ONNX
#
def load_tf(model_path: str):
    import tvm.relay.testing.tf as tf_testing

    # Creates graph from saved graph_def.pb.
    with tf_compat_v1.gfile.GFile(str(model_path), 'rb') as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    return graph_def, graph


def get_model_tf(model_path: str, input_shape=None, in_node: str = None, out_node: str = None, layout: str = 'NCHW'):
    """Load TF graph and converts it to Relay.
    """
    graph_def, graph = load_tf(model_path)

    if in_node and input_shape and out_node:
        mod, params = relay.frontend.from_tensorflow(graph_def,
                                                     layout=layout,
                                                     shape={in_node: input_shape},
                                                     outputs=[out_node])
    else:
        mod, params = relay.frontend.from_tensorflow(graph_def,
                                                     layout=layout)
    return mod, params, graph


def get_model_onnx(model_url: str, model_name: str, input_name: str = "1", input_shape=()):
    """Load an ONNX model."""
    import onnx
    model_path = download_testdata(model_url, model_name + '.onnx', module='onnx')
    onnx_model = onnx.load(model_path)

    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    return mod, params


def get_pretrained_model_pytorch(model_name: str, input_name: str, input_data: np.array):
    """Load a pretrained model from PyTorch zoo."""
    import torch
    import torchvision

    model = getattr(torchvision.models, model_name)(pretrained=True)

    # We grab the TorchScripted model via tracing
    scripted_model = torch.jit.trace(model, input_data).eval()

    shape_list = [(input_name, input_data.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    return mod, params


#
# TVM runtime
#
def generate_random_input_data(func: relay.Expr, mini=-1, maxi=1, ctx=tvm.cpu()):
    """Generates random values in [mini,maxi] for all "static" arguments of func except (dynamic) data
    that the application should provide.
    """
    net = run_infer_type(func)  # Infer Type aka params of function

    # init all tensors with random values
    shape_dict = {v.name_hint: v.checked_type for v in net.params}
    params = {}
    for k, v in shape_dict.items():
        if k == "data":
            continue

        init_value = np.random.uniform(mini, maxi, v.concrete_shape).astype(v.dtype)
        params[k] = tvm.nd.array(init_value, ctx=ctx)
    return params


def relay_test(mod: tvm.IRModule, params, input_var_name: str = 'data', data=None, target: str = 'llvm',
               host: str = 'llvm',
               ctx=tvm.cpu()):
    """Runs a TVM module on target with input data and model params.
    Returns output as tvm.ndarray
    """
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, target, params=params)

    return tvm_infer(graph, lib, params, input_var_name, data, ctx)


def tvm_infer(graph, lib, params, input_var_name: str = 'data', data=None, ctx=tvm.cpu()):
    """Runs a compiled TVM model (JSON graph, compiled module, tensors params) on input data"""
    module = runtime.create(graph, lib, ctx)
    module.set_input(input_var_name, data)
    module.set_input(**params)
    module.run()
    output = module.get_output(0)

    return output


def simplify_graph(mod, params=None):
    """Simplify module for inference on MLA."""
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.InferType(),  # these next 4 remove BN
                                    relay.transform.SimplifyInference(),
                                    relay.transform.FoldConstant(),
                                    relay.transform.FoldScaleAxis(),
                                    relay.transform.EliminateCommonSubexpr(),
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


def save_module(model_name, mod, graph=None, params=None, lib=None):
    with open(f"{model_name}_ir.txt", 'w') as f:
        f.write(str(mod['main']))

    if graph:
        with open(f"{model_name}_graph.json", "w") as f:
            f.write(graph)

    if params:
        with gzip.GzipFile(mode='wb', filename=f"{model_name}_params.gz") as f:
            f.write(relay.save_param_dict(params))

    if lib:
        lib.export_library(f"{model_name}_bin.so")


def load_module(model_name, mod, params, lib=None):
    loaded_json = open(f"{model_name}_graph.json").read()
    loaded_lib = tvm.runtime.load_module(f"{model_name}_bin.so")

    with gzip.GzipFile(mode='rb', filename=f"{model_name}_params.gz") as f:
        loaded_params = bytearray(f.read())

    return loaded_json, loaded_lib, loaded_params
