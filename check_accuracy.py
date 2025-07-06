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

device = "cpu"

transform_val = transforms.Compose([
    transforms.Resize((24, 94), interpolation=transforms.InterpolationMode.BICUBIC),     # сохранить высоту ширину — пропорционально
    transforms.ToTensor(),             # преобразование в тензор
        transforms.Normalize(mean=[0.496, 0.502, 0.504], std=[0.254, 0.2552, 0.2508])
])


ck_file = "./checkpoints/checkpoint_epoch_24.ckpt"

model = LPRNet(class_num=len(CHARS), dropout_prob = 0.5, out_indices=(2, 6, 13, 22))

# checkpoint = torch.load(ck_file, map_location=torch.device(device))
# model.load_state_dict(checkpoint['model_dict'])
model.load_state_dict(torch.load('lpr_best.pth'))


model.eval()
model.to(device)

STN = STNet()
STN.to(device)
# checkpoint = torch.load(ck_file, map_location=torch.device(device))
# STN.load_state_dict(checkpoint['stn_dict'])
STN.load_state_dict(torch.load('stn_best.pth'))
STN.eval()

print("Successful to build network!")

since = time.time()


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

dir = './val/img'

total_files = len(os.listdir(dir))
correct = 0

total_char_len = 0
total_char_correct = 0

progress_bar = tqdm(os.listdir(dir), desc="accuracy", leave=False)

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
    #     else:
    #         print(i, j)

print(f"accuracy: {correct / total_files:.4f}")
print(f"char accuracy: {total_char_correct / total_char_len:.4f}")
