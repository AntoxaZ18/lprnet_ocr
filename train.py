import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import collate_fn
from dataset import CHARS, PlateDataset
from model.LPRNET import LPRNet
from model.STN import STNet


def set_seed(seed=42):
    torch.manual_seed(seed)  # Фиксируем seed для CPU
    torch.cuda.manual_seed_all(seed)  # Для GPU
    np.random.seed(seed)
    random.seed(seed)


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


def sparse_tuple_for_ctc(lengths, T_length):
    input_lengths = []
    target_lengths = []

    for length in lengths:
        input_lengths.append(T_length)
        target_lengths.append(length)

    return torch.tensor(input_lengths, dtype=torch.int32), torch.tensor(
        target_lengths, dtype=torch.int32
    )


def flatten_labels(labels, lengths):
    flat_labels = []
    for label, length in zip(labels, lengths):
        flat_labels.extend(label[:length].tolist())
    return torch.tensor(flat_labels, dtype=torch.int)


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
                size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long
            ).to(device)

            target_lengths = torch.sum(targets != 0, dim=1)

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            total_loss += loss.item() * images.size(0)

    avg_val_loss = total_loss / len(val_loader.dataset)
    return avg_val_loss


def validate_model(model, dataloader, criterion, device):
    model.eval()
    STN.eval()
    running_loss = 0.0
    T_length = 18

    bar = tqdm(dataloader)

    with torch.no_grad():
        for images, labels, lengths in bar:
            images = images.to(device)
            flat_targets = labels.to(device)

            transfer = STN(images)
            logits = model(
                transfer
            )  # torch.Size([batch_size, CHARS length, output length ])
            # print("logit shapes", logits.shape)
            log_probs = logits.permute(2, 0, 1).log_softmax(
                2
            )  # for ctc loss: length of output x batch x length of chars

            input_lengths, target_lengths = sparse_tuple_for_ctc(
                lengths, T_length
            )  # convert to tuple with length as batch_size

            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            loss = criterion(
                log_probs,
                flat_targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )
            bar.set_postfix({'val loss': loss.item()})
            running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / len(dataloader.dataset)

    print(f"val loss: {avg_loss}")

    return avg_loss


def train_loop(
    model,
    STN,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    start_epoch,
    total_epochs,
    checkpoint_path,
):
    best_loss = float("inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    STN.to(device)

    train_losses = []
    validate_losses = []

    for epoch in range(start_epoch, total_epochs):
        train_loss = train(model, STN, criterion, optimizer, train_loader, device)
        val_loss = validate_model(model, val_loader, criterion, device)

        scheduler.step()
        print(f"Epoch {epoch} Training Loss: {train_loss:.3f} Val Loss: {val_loss:.3f}")

        train_losses.append(train_loss)
        validate_losses.append(val_loss)

        checkpoint = {
            "model_dict": model.state_dict(),
            "stn_dict": STN.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }

        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_epoch_{epoch}.ckpt")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "lpr_best.pth")
            torch.save(STN.state_dict(), "stn_best.pth")


def create_dataloader(
    image_dir: str = None, batch_size: int = 8, transforms=None, shuffle: bool = True
) -> DataLoader:
    """
    create dataloader
    """

    if transforms is None or image_dir is None:
        raise ValueError("trainsforms and image_dir must be set")

    dataset = PlateDataset(root_dir=image_dir, transform=transforms)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def load_checkpoint(ck_path: str, train_continue: bool = True):
    """
    ck_path - путь до папки с чекпоинтами
    create or load trainig state (models, optimizer, scheduler, epoch)
    """

    if os.path.exists(ck_path) and os.listdir(ck_path) and train_continue:
        last_chekpoint = sorted(
            os.listdir(ck_path), key=lambda x: int(x.split(".")[0].split("_")[-1])
        )[-1]
        checkpoint = torch.load(f"{ck_path}/{last_chekpoint}")
        return checkpoint
    else:
        os.makedirs(ck_path, exist_ok=True)

        return None


def initialize_from_ck(checkpoint, model, STN, optimizer, scheduler):
    model.load_state_dict(checkpoint["model_dict"])
    STN.load_state_dict(checkpoint["stn_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]


def train(model, STN, criterion, optimizer, dataloader, device):
    running_loss = 0.0
    T_length = 18

    for images, labels, lengths in (progress_bar := tqdm(dataloader, leave=False)):
        images = images.to(device)
        flat_targets = labels.to(device)

        transfer = STN(images)
        logits = model(transfer)
        log_probs = logits.permute(2, 0, 1).log_softmax(
            2
        )  # for ctc loss: length of output x batch x length of chars

        ctc_input_lengths, ctc_target_lengths = sparse_tuple_for_ctc(
            lengths, T_length
        )  # convert to tuple with length as batch_size

        ctc_input_lengths = ctc_input_lengths.to(device)
        ctc_target_lengths = ctc_target_lengths.to(device)

        loss = criterion(
            log_probs,
            flat_targets,
            input_lengths=ctc_input_lengths,
            target_lengths=ctc_target_lengths,
        )

        progress_bar.set_postfix({'train loss': loss.item()})


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument(
        "--proceed", type=bool, help="True for continue from saved checkpoint"
    )
    parser.add_argument("--batch", type=int, help="batchsize", default=128)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=100)

    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epochs
    proceed = args.proceed
    epochs = args.epochs
    lr = args.lr

    set_seed(42)

    mean = [0.496, 0.502, 0.504]
    std = [0.254, 0.2552, 0.2508]

    image_size = (24, 94)

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size,
                antialias=True,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomAffine(degrees=(-5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),  # сохранить высоту ширину — пропорционально
            transforms.ToTensor(),  # преобразование в тензор
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataloader = create_dataloader(
        image_dir="train/img", transforms=train_transform, batch_size=batch_size
    )
    val_dataloader = create_dataloader(
        image_dir="val/img",
        transforms=val_transform,
        batch_size=batch_size,
        shuffle=False,
    )

    model = LPRNet(class_num=len(CHARS), dropout_prob=0.5, out_indices=(2, 6, 13, 22))
    STN = STNet()
    criterion = nn.CTCLoss(blank=len(CHARS) - 1, reduction="mean")

    optimizer = torch.optim.Adam(
        [
            {"params": STN.parameters(), "weight_decay": 2e-5},
            {"params": model.parameters()},
        ],
        lr=0.01,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=epochs
    )

    CHECKPOINT_PATH = "./checkpoints"
    checkpoint = load_checkpoint(CHECKPOINT_PATH, train_continue=proceed)

    if checkpoint is not None:
        start_epoch = initialize_from_ck(checkpoint, model, STN, optimizer, scheduler)
    else:
        start_epoch = 0

    train_loop(
        model,
        STN,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        scheduler,
        start_epoch,
        epochs,
        CHECKPOINT_PATH,
    )
