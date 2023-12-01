import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from utils.utils import prepare_model

class PreProcess(nn.Module):
    """Normalize tensor
       Expected batch x channels x height x width
    """

    def __init__(self) -> None:
        super(PreProcess, self).__init__()
        self.normalzie = transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))

    def forward(self, x):
        y = x / 255
        y = self.normalzie(y)
        return y

def export_model(args):
    """Export pth model to onnx.
       This function will create only batch size 1 model.
       Not supported dynamic batch.

    Args:
        args (argparse): Argsparse object contains weight path and output onnx path.
    """

    # batch, channels, height, width
    x = torch.randn(1, 3, 28, 28)
    model = prepare_model(args.pth_path)
    whole_model = nn.Sequential(
        PreProcess(),
        model
    )
    whole_model.eval()
    print("Start exporting")
    torch.onnx.export(
        whole_model,
        x,
        args.output_path,
        export_params=True,
        opset_version=13,
        input_names=["input"],
        output_names=["output"])
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth-path', type=str, default='best.pth', help='weights path to be exported')
    parser.add_argument('--output-path', type=str, default='best.onnx', help='output onnx path')
    args = parser.parse_args()
    export_model(args)
    