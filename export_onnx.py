import argparse

import torch
from onnxoptimizer import optimize
from onnxsim import simplify

import onnx
from dataset import CHARS
from model.LPRNET import LPRNet
from model.STN import STNet


class STNLPRNet(torch.nn.Module):
    """
    class for fusing stn + lpr model
    """

    def __init__(self, stn, lpr):
        super().__init__()
        self.stn = stn
        self.lpr = lpr

    def forward(self, x):
        transformed = self.stn(x)
        result = self.lpr(transformed)
        return result


def export(stn_checkpoint: str, lpr_checkpoint: str, output_name: str) -> None:
    lpr = LPRNet(class_num=len(CHARS), dropout_prob=0.5, out_indices=(2, 6, 13, 22))
    stn = STNet()

    lpr.load_state_dict(torch.load(lpr_checkpoint))
    stn.load_state_dict(torch.load(stn_checkpoint))

    stn_lprnet = STNLPRNet(stn, lpr)
    stn_lprnet.eval()

    example_inputs = torch.randn(1, 3, 24, 94)
    # Экспорт в ONNX
    torch.onnx.export(
        stn_lprnet,  # модель
        example_inputs,  # пример входа
        output_name,  # путь к файлу
        export_params=True,  # сохранить веса в модели
        input_names=["input"],  # имена входов
        output_names=["output"],  # имена выходов
        opset_version=20,  # for affine transform
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def optimize_onnx(model_path: str, optimized_model_path: str):
    model = onnx.load(model_path)

    # Оптимизация
    optimized_model = optimize(model)

    # Упрощение
    optimized_model, _ = simplify(optimized_model)

    onnx.save(model, optimized_model_path)


def validate(model_path: str):
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("exported model is ok")
    except Exception as e:
        print("exported model error:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stn", type=str, help="path to stn model")
    parser.add_argument("--lpr", type=str, help="path to lpr model")
    parser.add_argument("--onnx", type=str, help="path to fused onnx model")

    args = parser.parse_args()

    stn_model = args.stn
    lpr_model = args.lpr
    onnx_model = args.out

    print(
        f"Models '{stn_model}' and '{lpr_model}' will be fused and exported to '{onnx_model}'"
    )

    export(stn_model, lpr_model, onnx_model)
    optimize_onnx(onnx_model, onnx_model)
    validate(onnx_model)
