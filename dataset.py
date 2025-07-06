import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random

CHARS = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
     'Y', 'X', '-'
]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}


def collate_fn(batch):
    images, labels = zip(*batch)

    # Собираем батч
    images_tensor = torch.stack(images, dim=0)  # [B, C, H, W]
    labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    return images_tensor, labels_tensor

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
        label = img_name.split('.')[0]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        # Преобразуем label в список индексов
        encoded_label = [CHARS_DICT[c] for c in label]
        return image, encoded_label, len(encoded_label)