import os
import torch
import torchvision
from net_v6 import BidirectionalRestorer_V6
from meg_2_tf import load_from_meg
import argparse
import numpy as np

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
    inputs2 = torch.zeros((5, 30, 180, 320))
    netG.eval()
    netG(inputs2)

    print("converting to mobile....")
    traced_model = torch.jit.trace(netG, inputs2)
    traced_model_optimized = torch.utils.mobile_optimizer.optimize_for_mobile(
        traced_model)
    save_path = "model.pt"
    torch.jit.save(traced_model_optimized, save_path)

    print("done!")
