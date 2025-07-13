import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CHARS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "E",
    "K",
    "M",
    "H",
    "O",
    "P",
    "C",
    "T",
    "Y",
    "X",
    "-",
]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(img)
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


class PlateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.files = os.listdir(self.root_dir)
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        label = img_name.split(".")[0]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        # Преобразуем label в список индексов
        encoded_label = [CHARS_DICT[c] for c in label]
        return image, encoded_label, len(encoded_label)
