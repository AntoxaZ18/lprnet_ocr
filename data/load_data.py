import torch
from torch.utils.data import *
# from imutils import paths
import numpy as np
import random
# import cv2
import os


        
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
        
# if __name__ == "__main__":
    
#     dataset = LPRDataLoader(['validation'], (94, 24))   
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn)
#     print('data length is {}'.format(len(dataset)))
#     for imgs, labels, lengths in dataloader:
#         print('image batch shape is', imgs.shape)
#         print('label batch shape is', labels.shape)
#         print('label length is', len(lengths))      
#         break
    
