import torch
from PIL import Image
from model.LPRNET import LPRNet
from model.STN import STNet
import time
import os
from utils import decode_function, BeamDecoder
from torchvision import transforms
from dataset import CHARS
from tqdm import tqdm
import argparse


def test_model(stn: str, lpr: str, dir: str):

    '''
    Тест метрик модели на изображениях из dir
    '''
    device = "cpu"

    transform_val = transforms.Compose([
        transforms.Resize((24, 94), interpolation=transforms.InterpolationMode.BICUBIC),     # сохранить высоту ширину — пропорционально
        transforms.ToTensor(),             # преобразование в тензор
        transforms.Normalize(mean=[0.496, 0.502, 0.504], std=[0.254, 0.2552, 0.2508])
    ])

    model = LPRNet(class_num=len(CHARS), dropout_prob = 0.5, out_indices=(2, 6, 13, 22))
    model.load_state_dict(torch.load(lpr))
    model.to(device)
    model.eval()

    STN = STNet()
    STN.load_state_dict(torch.load(stn))
    STN.to(device)
    STN.eval()


    def prepare(image):
        data = transform_val(image)
        data = data.unsqueeze(0).to(device)
        return data

    def predict(tensor):
        transfer = STN(tensor)
        predictions = model(transfer)
        predictions = predictions.cpu().detach().numpy()  # (1, 68, 18)
        labels, prob, pred_labels = decode_function(predictions, CHARS, BeamDecoder)

        return labels[0]


    total_files = len(os.listdir(dir))

    correct = 0
    total_char_len = 0
    total_char_correct = 0

    progress_bar = tqdm(os.listdir(dir), desc="accuracy bench", leave=False)

    for image in progress_bar:

        true_label = image.split('.')[0]

        total_char_len += len(true_label)

        image = Image.open(f"{dir}/{image}")

        tensor = prepare(image)
        predicted = predict(tensor)

        if (true_label == predicted):
            correct += 1

        for i, j in zip(true_label, predicted):
            if i == j:
                total_char_correct += 1

    return correct / total_files, total_char_correct / total_char_len



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stn", type=str, help="path to stn")
    parser.add_argument("--lpr", type=str, help="path to lpr")
    parser.add_argument("--images", type=str, help="images dir")

    args = parser.parse_args()

    stn = args.stn
    lpr = args.lpr
    images = args.images

    print(f"model {stn} + {lpr} will be tested on '{images}' folder")

    word_accuracy, char_accuracy = test_model(stn, lpr, images)

    print(f"word accuracy: {word_accuracy * 100:.4f} %")
    print(f"char accuracy: {char_accuracy * 100:.4f} %")







