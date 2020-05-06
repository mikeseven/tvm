import gzip
import os

import numpy as np
import tvm
from tvm import relay
from tvm.relay.sima import qop

from tvm.relay.testing import run_opt_pass

from relay_viz import relay_viz
from tqdm.auto import tqdm
import logging

import tensorflow as tf

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf


def get_calibration_dataset(input_name="data", shape=(1, 16, 64, 64), mini=0, maxi=255, num=10, dtype='float32'):
    dataset = []
    for i in range(num):
        data = np.random.uniform(mini, maxi, size=shape).astype(dtype)
        dataset.append({input_name: data})
    return dataset


def conv_add_relu():
    data = relay.var("data", shape=(1, 16, 64, 64))  # NCHW
    weight = relay.var("weight")
    conv = relay.nn.conv2d(data, weight,
                           kernel_size=(3, 3),
                           padding=(1, 1),
                           channels=16)

    # Note: using nn.bias_add is NOT recognized by TVM auto-quantize
    bias = relay.var("bias", shape=(1, 16, 1, 1))  # vector of NxOx1x1 (must be 4D here, no broadcasting)
    biasAdd = relay.add(conv, bias)
    y = relay.nn.relu(biasAdd)

    return y


def _load_tf_model(model_path: str, input_shape, in_node: str, out_node: str, input_dtype: str = 'uint8',
                   layout: str = None):
    """Load TF graph and converts it to Relay.
    for GPU, use layout='NCHW'
    """

    def _load_tf(model_path: str):
        # Tensorflow utility functions
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

    graph_def, graph = _load_tf(model_path)
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                                 layout=layout,
                                                 shape={in_node: input_shape},
                                                 outputs=[out_node])

    return mod, params, graph


def load_bosch_model():
    home_dir = os.path.expanduser('~')
    model_base = os.path.join(home_dir, 'Downloads/Bosch_DL_HW_Benchmark/03_benchmark')
    model_repo = os.path.join(model_base, '01_feature_extractors/02_small_vgg')
    model_name = 'small_vgg_tf_graph'
    model_path = os.path.join(model_repo, model_name + '.pb')

    in_node = 'image'
    out_node = 'conv5_3/Relu'
    start_name = 'nn.conv2d'
    stop_name = 'nn.softmax'
    input_shape = (1, 1080, 1920, 3)
    image_repo = os.path.join(home_dir, 'cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/cologne')

    return _load_tf_model(model_path, input_shape, in_node=in_node, out_node=out_node)


def main():
    """main driver for SiMa quantization"""

    # setup TVM
    host = "llvm -mcpu=skylake"
    target = host
    ctx = tvm.cpu()  # tvm.context(target)

    model_name = 'sima_quant_test'
    temp_dir = os.path.join(os.path.expanduser('~'), model_name)

    try:
        os.mkdir(temp_dir)
    except:
        pass

    model = conv_add_relu()
    params = None

    # quantization hyper-params
    # - 8-bit power2 linear quantization
    # - asymmetric for activations
    # - signed quantized values
    qchild = qop.QOp(model)
    qgraph = qop.QGraph(model, B=8, sym=False, signed=True)

    qgraph.analyze()

    weight_val = np.random.uniform(size=(16, 16, 3, 3), low=-10, high=10).astype('float32')
    bias_val = np.random.uniform(size=(1, 16, 1, 1), low=-1, high=1).astype('float32')

    # TODO note: for each child we must provide its params other than input
    params = {
        'weight': weight_val,
        'bias': bias_val,
    }
    qchild.set_params(params)
    num_data = 10
    dataset = get_calibration_dataset("data", shape=(1, 16, 64, 64), mini=0, maxi=255, num=num_data, dtype='float32')
    for data in tqdm(dataset, total=num_data):
        qchild.calibrate(data['data'], ctx, target)
    qchild.realize()

    print(qchild)
    # print(qchild.astext())

    print(qgraph.astext())

if __name__ == "__main__":
    logging.getLogger("autotvm").setLevel(logging.CRITICAL)
    main()
