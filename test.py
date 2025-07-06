import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import random
import numpy as np
from model.LPRNET import LPRNet
from model.STN import STNet
from data.load_data import collate_fn
from dataset import PlateDataset
from torchvision import transforms
from dataset import CHARS
import os

def weights_initialization(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



def initialize_lprnet_weights(lpr_model):
    """Initialize weights for LPR net"""
    lpr_model.backbone.apply(weights_initialization)
    lpr_model.container.apply(weights_initialization)
    print('Successful init LPR weights')

def set_seed(seed=42):
    torch.manual_seed(seed)             # Фиксируем seed для CPU
    torch.cuda.manual_seed_all(seed)    # Для GPUclar
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)
total_iters = 0
BATCH_SIZE=128

# # Параметры
imgH = 24
imgw = 94


transform_train = transforms.Compose([
    transforms.Resize((imgH, imgw), antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.7),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(degrees=(-5, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.496, 0.502, 0.504], std=[0.254, 0.2552, 0.2508])
])

# # Трансформации и загрузчик данных
transform_val = transforms.Compose([
    transforms.Resize((imgH, imgw), interpolation=transforms.InterpolationMode.BICUBIC),     # сохранить высоту ширину — пропорционально
    transforms.ToTensor(),             # преобразование в тензор
    transforms.Normalize(mean=[0.496, 0.502, 0.504], std=[0.254, 0.2552, 0.2508]),
])


train_dataset = PlateDataset(
    root_dir="train/img/",
    transform=transform_train
)

val_dataset = PlateDataset(
    root_dir="val/img/",
    transform=transform_val
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_mean_std(loader):
    channels_sum, channels_squared_sum = 0, 0
    num_samples = 0

    for images, _, _ in loader:
        N, C, H, W = images.shape
        num_samples += N * H * W  # общее количество пикселей

        data = images.view(N, C, -1)  # (N, C, H*W)

        channels_sum += data.sum(2).sum(0)  # (C,)
        channels_squared_sum += (data.pow(2)).sum(2).sum(0)  # (C,)

    total_mean = channels_sum / num_samples
    total_var = channels_squared_sum / num_samples - total_mean.pow(2)
    total_std = torch.sqrt(total_var)

    print("Mean:", total_mean.tolist())
    print("Std:", total_std.tolist())

    return total_mean, total_std


# calculate_mean_std(train_loader)


num_epochs = 80
start_epoch = 0


# model = CRNN(imgH=imgH, nc=nc, nclass=num_classes, nh=nh)
model = LPRNet(class_num=len(CHARS), dropout_prob=0.5, out_indices=(2, 6, 13, 22))
# initialize_lprnet_weights(model)
model.to(device)

STN = STNet()
STN.to(device)

lr = 2e-3

optimizer = torch.optim.Adam([{'params': STN.parameters(), 'weight_decay': 2e-5},
                                {'params': model.parameters()}], lr=lr)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs
)

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

CHECKPOINT_PATH = './checkpoints'

if os.path.exists(CHECKPOINT_PATH) and os.listdir(CHECKPOINT_PATH):
    last_chekpoint = sorted(os.listdir(CHECKPOINT_PATH), key=lambda x: int(x.split('.')[0].split('_')[-1]))[-1]
    ck = torch.load(f"{CHECKPOINT_PATH}/{last_chekpoint}")
    model.load_state_dict(ck['model_dict'])
    STN.load_state_dict(ck['stn_dict'])
    optimizer.load_state_dict(ck['optimizer_state_dict'])
    scheduler.load_state_dict(ck['scheduler_state_dict'])
    start_epoch = ck['epoch'] + 1
    print(f"Restore from checkpoint, continue from epoch {start_epoch}")
else:
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

criterion = nn.CTCLoss(blank=len(CHARS) - 1, reduction="mean") 

print('start', start_epoch)


def sparse_tuple_for_ctc(lengths, T_length):
    input_lengths = []
    target_lengths = []

    for length in lengths:
        input_lengths.append(T_length)
        target_lengths.append(length)

    return torch.tensor(input_lengths, dtype=torch.int32), torch.tensor(target_lengths, dtype=torch.int32)



def flatten_labels(labels, lengths):
    flat_labels = []
    for label, length in zip(labels, lengths):
        flat_labels.extend(label[:length].tolist())
    return torch.tensor(flat_labels, dtype=torch.int)




def train_model(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    STN.train()
    
    running_loss = 0.0
    global total_iters

    # Оборачиваем dataloader в tqdm для отображения прогресса
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
    T_length = 18

    for images, labels, lengths in progress_bar:
        images = images.to(device)
        flat_targets = labels.to(device)

        transfer = STN(images)
        logits = model(transfer)  # torch.Size([batch_size, CHARS length, output length ])
        # print("logit shapes", logits.shape)
        log_probs = logits.permute(2, 0, 1).log_softmax(2) # for ctc loss: length of output x batch x length of chars
        
        ctc_input_lengths, ctc_target_lengths = sparse_tuple_for_ctc(lengths, T_length) # convert to tuple with length as batch_size 

        # if (flat_targets == 0).any():
        #     print("Warning: targets contains blank token (0), which is not allowed")

        ctc_input_lengths = ctc_input_lengths.to(device)
        ctc_target_lengths = ctc_target_lengths.to(device)
    
        loss = criterion(log_probs, flat_targets, input_lengths=ctc_input_lengths, target_lengths=ctc_target_lengths)
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 200 or loss.item() < -200:
            print("log_probs contains NaN:", torch.isnan(log_probs).any().item())
            print("log_probs contains Inf:", torch.isinf(log_probs).any().item())
            print(f"Loss is bad {loss.item()}")
            print(flat_targets)
            print(ctc_input_lengths, ctc_target_lengths)
            # print(log_probs)
            # print(log_probs[:, 0, :].exp().sum(dim=1))  # сумма exp(log_probs) по классам должна быть ≈ 1
            import sys
            sys.exit()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # break

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch} Training Loss: {epoch_loss:.4f}")

def evaluate(model, val_loader, criterion, device):
    model.eval()
    STN.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            targets = labels.to(device)

            outputs = model(images)

            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)

            target_lengths = torch.sum(targets != 0, dim=1)

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            total_loss += loss.item() * images.size(0)

    avg_val_loss = total_loss / len(val_loader.dataset)
    return avg_val_loss

def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    T_length = 18

    with torch.no_grad():
        for images, labels, lengths in dataloader:
            images = images.to(device)
            flat_targets = labels.to(device)

            transfer = STN(images)
            logits = model(transfer)  # torch.Size([batch_size, CHARS length, output length ])
            # print("logit shapes", logits.shape)
            log_probs = logits.permute(2, 0, 1).log_softmax(2) # for ctc loss: length of output x batch x length of chars
            
            input_lengths, target_lengths = sparse_tuple_for_ctc(lengths, T_length) # convert to tuple with length as batch_size 

            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
    
            loss = criterion(log_probs, flat_targets, input_lengths=input_lengths, target_lengths=target_lengths)
            running_loss += loss.item() * images.size(0)


    avg_loss = running_loss / len(dataloader.dataset)

    print(f'val loss: {avg_loss}')

    return avg_loss


best_loss = None

# idx = 500

# print(torch.max(train_dataset[idx][0]))

# x = train_dataset[0][0]
# x = x.unsqueeze(0)
# print('x shape', x.shape)

# x = x.to(device)
# model.eval()
# y = model(x)

# print('output shape', y.shape)


# import sys
# sys.exit(0)


# import sys
# sys.exit(0)

# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

model.to(device)
for epoch in range(start_epoch, num_epochs):  # обучение на 20 эпохах
    train_model(model, train_loader, criterion, optimizer, device, epoch)
    val_loss = validate_model(model, val_loader, criterion, device)
    scheduler.step()  
    # lr_sheduler.step()
    
    checkpoint = {
        'model_dict': model.state_dict(),
        'stn_dict': STN.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }

    torch.save(checkpoint, f'./checkpoints/checkpoint_epoch_{epoch}.ckpt')

    if best_loss is None:
        best_loss = val_loss

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'lpr_best.pth')
        torch.save(STN.state_dict(), 'stn_best.pth')





                # total_iters += 1
            # if total_iters % 100 == 0:
            #     # current training accuracy             
            #     preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
            #     _, pred_labels = decode(preds, CHARS)  # list of predict output
            #     total = preds.shape[0]
            #     start = 0
            #     TP = 0
            #     for i, length in enumerate(lengths):
            #         label = labels[start:start+length]
            #         start += length
            #         if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
            #             TP += 1
                
            #     print(f"accuracy: {TP/total}" )