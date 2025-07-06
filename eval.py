import torch
from PIL import Image
from model.LPRNET import LPRNet
from model.STN import STNet
import time
import numpy as np
from utils import decode_function, BeamDecoder
from torchvision import transforms
from dataset import CHARS
import cv2


device = "cpu"

transform_val = transforms.Compose([
    transforms.Resize((24, 94), interpolation=transforms.InterpolationMode.BICUBIC),     # сохранить высоту ширину — пропорционально
    transforms.ToTensor(),             # преобразование в тензор
    transforms.Normalize(mean=[0.496, 0.502, 0.504], std=[0.254, 0.2552, 0.2508]),
])

mean = np.array([0.496, 0.502, 0.504], dtype=np.float32)
std = np.array([0.254, 0.2552, 0.2508], dtype=np.float32)

def batch_transform(rgb_batch, mean, std):
    """
    Обрабатывает батч изображений: resize, transpose, normalize.

    Параметры:
        rgb_batch (np.ndarray): Входной батч изображений в формате (B, H, W, C)
        mean (list or np.ndarray): Средние значения для каждого канала (C,)
        std (list or np.ndarray): Стандартные отклонения для каждого канала (C,)

    Возвращает:
        np.ndarray: Нормализованный батч в формате (B, C, H, W)
    """
    B, H, W, C = rgb_batch.shape

    # Конвертируем в float32 и делим на 255, если это uint8
    if rgb_batch.dtype == np.uint8:
        rgb_batch = rgb_batch.astype(np.float32) / 255.0

    # Resize (24, 94) — высота 24, ширина 94
    resized_batch = np.zeros((B, 24, 94, C), dtype=np.float32)
    for i in range(B):
        resized_batch[i] = cv2.resize(rgb_batch[i], (94, 24))  # (W, H)

    # Перевод в формат (B, C, H, W)
    transposed = np.transpose(resized_batch, (0, 3, 1, 2))  # (B, C, H, W)

    # Нормализация по каналам
    for i in range(transposed.shape[1]):  # для каждого канала
        transposed[:, i, :, :] = (transposed[:, i, :, :] - mean[i]) / std[i]

    contiguous_transposed = np.ascontiguousarray(transposed)

    return contiguous_transposed.astype(np.float32)


model = LPRNet(class_num=len(CHARS), dropout_prob = 0.5, out_indices=(2, 6, 13, 22))

ck_file = "./checkpoints/checkpoint_epoch_24.ckpt"


checkpoint = torch.load(ck_file, map_location=torch.device(device))
model.load_state_dict(checkpoint['model_dict'])

model.eval()
model.to(device)

STN = STNet()
STN.to(device)
checkpoint = torch.load(ck_file, map_location=torch.device(device))
STN.load_state_dict(checkpoint['stn_dict'])
STN.eval()

print("Successful to build network!")

since = time.time()
image = Image.open("cropped____.jpg")
# image = image.resize((94, 24))

# im = (np.transpose(np.float32(image), (2, 0, 1)) - 127.5) * 0.0078125
# data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94])

# data_trans = transform_val(image)
# data_trans = data_trans.unsqueeze(0).to(device)

data = batch_transform(np.expand_dims(image, 0), mean, std)
data = torch.from_numpy(data).to(device)

# print(data.shape)
# print(data_trans.shape)

# print(data[0][0])
# print(data_trans[0][0])



transfer = STN(data)
predictions = model(transfer)
predictions = predictions.cpu().detach().numpy()  # (1, 68, 18)
print(predictions.shape, predictions[0][0])

labels, prob, pred_labels = decode_function(predictions, CHARS, BeamDecoder)
print("model inference in {:2.3f} seconds".format(time.time() - since))

print(labels, prob, pred_labels)

# transformed_img = convert_output_image(transfer)
# pad_image = cv2.copyMakeBorder(transformed_img, top=15, bottom=0, left=0, right=0,
#                                 borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
if (prob[0] < -85) and (len(labels[0]) in [8, 9]):
    print(labels[0])
#     pad_image = add_text2image(pad_image, (labels[0]), TextPosition(16, 0), text_size=10)

# cv2.imshow('Prediction', pad_image)
# cv2.waitKey()
# cv2.destroyAllWindows()