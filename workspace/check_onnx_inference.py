import argparse
import random

import numpy as np
import onnxruntime
import torch

from utils.utils import create_dataset, prepare_model

def show_result(gt, pth_results, onnx_results):
        pth_results = pth_results.detach().numpy()
        max_index = np.argmax(pth_results, axis=1)
        print("***************")
        print(f"GT: {gt}")
        print(f"Prediction by Pytorch: Label = '{max_index.item()}', Score = '{pth_results[0, max_index].item()}'")
        max_index = np.argmax(onnx_results[0], axis=1)
        print(f"Prediction by ONNX   : Label = '{max_index[0].item()}', Score = '{onnx_results[0][0, max_index[0]].item()}'")
        print("***************\n")


def compare_inference_result(args):
    """Compare inference result of .pth and .onnx

    Args:
        args (argparse): argparse object contains pth path and onnx path
    """
    pth_path = args.pth_path
    onnx_path = args.onnx_path
    model = prepare_model(pth_model_path=pth_path)
    sess = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    model.eval()
    tensor_data = create_dataset(is_train=False, normalize=True, invert=False)
    pil_data = create_dataset(is_train=False, normalize=False, invert=False)
    infer_index_list = [random.randint(0, len(tensor_data)) for _ in range(3)]
    for image_idx in infer_index_list:
        tensor_image, label = tensor_data[image_idx]
        pil_image, _ = pil_data[image_idx]
        np_image = np.array(pil_image).astype(np.float32)
        np_image = np.transpose(np_image, (2, 0, 1))
        np_image = np.expand_dims(np_image, axis=0)
        tensor_image = tensor_image.unsqueeze(dim=0)
        pth_results = model(tensor_image)
        onnx_results = sess.run(["output"], {"input": np_image})
        show_result(label, pth_results, onnx_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth-path', type=str, default='best.pth', help='weights path to be exported')
    parser.add_argument('--onnx-path', type=str, default='best.onnx', help='output onnx path')
    args = parser.parse_args()
    compare_inference_result(args)