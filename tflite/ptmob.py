import os
import tensorflow as tf
from tensorflow import keras
from net_v6 import BidirectionalRestorer_V6
from meg_2_tf import load_from_meg
import argparse
import numpy as np
import onnx
import onnx_tf
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_args():
    parser = argparse.ArgumentParser(description='to get tflite')
    parser.add_argument('--mgepath', default=None, type=str,
                        help='the path to xxx.mge pretrained model')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    input_shape = (1, 180, 320, 30)

    netG = BidirectionalRestorer_V6()

    print("building ....")
    netG.build(input_shape)
    print("build ok!!")
    netG.summary()

    netG.compute_output_shape(input_shape=input_shape)

    netG = load_from_meg(netG, args.mgepath)

    print("testing...")
    inputs2 = np.zeros((5, 180, 320, 30))
    netG.predict(inputs2, batch_size=1, verbose=1)

    print("converting to onnx....")
    onnx_model_path = "model.onnx"
    tf.keras.models.save_model(netG, 'temp.h5', include_optimizer=False)
    model = keras.models.load_model('temp.h5', compile=False)
    onnx_model = onnx_tf.convert_keras(model, output_path=onnx_model_path)

    print("converting to pytorch....")
    pytorch_model = onnx.load(onnx_model_path)
    pytorch_model = onnx_tf.backend.prepare(pytorch_model, device='CPU')
    pytorch_model_path = "model.pt"
    torch.onnx.export(pytorch_model, inputs2, pytorch_model_path)

    print("converting to mobile....")
    traced_model = torch.jit.load(pytorch_model_path)
    traced_model_optimized = torch.jit.optimize(traced_model)
    traced_model_optimized._c._create_mobile_module()
    traced_model_optimized_path = "model_mobile.pt"
    torch.jit.save(traced_model_optimized, traced_model_optimized_path)

    print("done!")
