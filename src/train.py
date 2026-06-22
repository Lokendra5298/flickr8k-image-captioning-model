# src/train.py

import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm

from config import CFG
from dataset import build_loaders
from model import ImageCaptioningModel


def save_checkpoint(model, optimizer, vocab, epoch, loss, val_acc):
    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "val_acc": val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_stoi": vocab.stoi,
        "vocab_itos": vocab.itos,
        "vocab_min_freq": vocab.min_freq,
    }

    torch.save(checkpoint, CFG.CHECKPOINT_PATH)


def load_checkpoint_if_exists(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting training from scratch.")
        return 1, float("inf")

    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["loss"]

    print(f"Resuming from epoch {start_epoch}")
    print(f"Best validation loss so far: {best_val_loss:.4f}")

    return start_epoch, best_val_loss


def token_accuracy(predictions, targets, pad_idx):
    """
    Calculates word/token-level accuracy.

    predictions:
        [batch_size, seq_len, vocab_size]

    targets:
        [batch_size, seq_len]

    This ignores <pad> tokens.
    """
    predicted_tokens = predictions.argmax(dim=-1)

    non_pad_mask = targets != pad_idx

    correct = (
        (predicted_tokens == targets) &
        non_pad_mask
    ).sum().item()

    total = non_pad_mask.sum().item()

    if total == 0:
        return 0.0

    return correct / total


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, pad_idx):
    model.train()

    total_loss = 0.0
    total_acc = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for images, captions in progress_bar:
        images = images.to(device)
        captions = captions.to(device)

        input_captions = captions[:, :-1]
        target_captions = captions[:, 1:]

        predictions = model(images, input_captions)

        loss = criterion(
            predictions.reshape(-1, predictions.shape[-1]),
            target_captions.reshape(-1)
        )

        acc = token_accuracy(
            predictions=predictions,
            targets=target_captions,
            pad_idx=pad_idx
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc

        progress_bar.set_postfix(
            loss=loss.item(),
            acc=f"{acc * 100:.2f}%"
        )

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)

    return avg_loss, avg_acc


def evaluate(model, data_loader, criterion, device, pad_idx):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for images, captions in data_loader:
            images = images.to(device)
            captions = captions.to(device)

            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]

            predictions = model(images, input_captions)

            loss = criterion(
                predictions.reshape(-1, predictions.shape[-1]),
                target_captions.reshape(-1)
            )

            acc = token_accuracy(
                predictions=predictions,
                targets=target_captions,
                pad_idx=pad_idx
            )

            total_loss += loss.item()
            total_acc += acc

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_acc / len(data_loader)

    return avg_loss, avg_acc


def write_log_header_if_needed():
    os.makedirs(CFG.LOG_DIR, exist_ok=True)

    if not os.path.exists(CFG.TRAIN_LOG_PATH):
        with open(CFG.TRAIN_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "test_loss",
                "train_acc",
                "val_acc",
                "test_acc",
            ])


def append_log(epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc):
    write_log_header_if_needed()

    with open(CFG.TRAIN_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            test_loss,
            train_acc,
            val_acc,
            test_acc,
        ])


def main():
    device = torch.device(CFG.DEVICE)

    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, vocab = build_loaders()

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    model = ImageCaptioningModel(
        embed_size=CFG.EMBED_SIZE,
        hidden_size=CFG.HIDDEN_SIZE,
        vocab_size=len(vocab),
        num_layers=CFG.NUM_LAYERS,
        dropout=CFG.DROPOUT
    ).to(device)

    pad_idx = vocab.stoi[vocab.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.LEARNING_RATE
    )

    start_epoch, best_val_loss = load_checkpoint_if_exists(
        model=model,
        optimizer=optimizer,
        checkpoint_path=CFG.CHECKPOINT_PATH,
        device=device
    )

    for epoch in range(start_epoch, CFG.NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            pad_idx=pad_idx
        )

        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            pad_idx=pad_idx
        )

        test_loss, test_acc = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            pad_idx=pad_idx
        )

        print(
            f"Epoch [{epoch}/{CFG.NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Train Acc: {train_acc * 100:.2f}% "
            f"Val Acc: {val_acc * 100:.2f}% "
            f"Test Acc: {test_acc * 100:.2f}%"
        )

        append_log(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=test_acc
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                vocab=vocab,
                epoch=epoch,
                loss=val_loss,
                val_acc=val_acc
            )

            print("Saved best checkpoint.")


if __name__ == "__main__":
    main()