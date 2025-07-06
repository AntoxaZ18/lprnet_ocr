
import torch
from model.STN import STNet
from model.LPRNET import LPRNet
from old.network_torch import CRNN
from dataset import CHARS
import onnx
from onnxoptimizer import optimize
from onnxsim  import simplify

from torchvision import transforms
device = "cpu"

transform_val = transforms.Compose([
    transforms.Resize((32, 94), interpolation=transforms.InterpolationMode.BICUBIC),     # сохранить высоту ширину — пропорционально
    transforms.ToTensor(),             # преобразование в тензор
    transforms.Normalize(mean=[0.496, 0.502, 0.504], std=[0.254, 0.2552, 0.2508]),
])


class STNLPRNet(torch.nn.Module):
    def __init__(self, stn, lpr):
        super().__init__()
        self.stn = stn
        self.lpr = lpr

    def forward(self, x):
        transformed = self.stn(x)
        result = self.lpr(transformed)
        return result


def export(stn_checkpoint: str, lpr_checkpoint: str, output_name: str) -> None:

    lpr = LPRNet(class_num=len(CHARS), dropout_prob = 0.5, out_indices=(2, 6, 13, 22))
    stn = STNet()

    lpr.load_state_dict(torch.load(lpr_checkpoint))
    stn.load_state_dict(torch.load(stn_checkpoint))

    stn_lprnet = STNLPRNet(stn, lpr)
    stn_lprnet.eval()

    example_inputs = (torch.randn(1, 3, 24, 94))
    # Экспорт в ONNX
    torch.onnx.export(
        stn_lprnet,                # модель
        example_inputs,            # пример входа
        output_name,            # путь к файлу
        export_params=True,        # сохранить веса в модели
        input_names=["input"],     # имена входов
        output_names=["output"],   # имена выходов
        opset_version=20,   #for affine transform
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

def optimize_onnx(model_path:str, optimized_model_path: str):
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
        print("Модель корректна.")
    except Exception as e:
        print("Ошибка в модели:", e)


ONNX_MODEL = "stn_lpr.onnx"
OPTIMIZED_MODEL = "stn_lpr_opt_2.onnx"

export("stn_best.pth", "lpr_best.pth", ONNX_MODEL)
optimize_onnx(ONNX_MODEL, OPTIMIZED_MODEL)
validate(OPTIMIZED_MODEL)

