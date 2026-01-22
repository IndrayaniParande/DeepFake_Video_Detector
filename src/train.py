import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split



#Create Train & Validation DataLoaders

def train_val_loaders(
    dataset,
    batch_size=8,
    split_ratio=0.8,
    num_workers=2
):
    """
    Splits dataset into train and validation sets
    and returns corresponding DataLoaders.
    """

    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"Train clips: {len(train_ds)} | Val clips: {len(val_ds)}")

    return train_loader, val_loader, train_ds, val_ds



#Train One Epoch

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in train_loader:
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = 100.0 * correct / total

    return avg_loss, acc



# Full Model Trainining

def train_model(
    model,
    train_loader,
    device,
    epochs=25,
    lr=1e-4,
    class_weights=(1.0, 1.3),
    step_size=8,
    gamma=0.5
):
    
    weights = torch.tensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

        scheduler.step()



# Exporting Model to ONNX

def export_model_to_onnx(
    model,
    onnx_path,
    device,
    clip_length=16,
    height=112,
    width=112
):
    dummy_input = torch.randn(
        1,
        3,
        clip_length,
        height,
        width
    ).to(device)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"}
        }
    )

    print("ONNX model exported to:")
    print(onnx_path)

if __name__ == "__main__":
    print("Model trained successfully")