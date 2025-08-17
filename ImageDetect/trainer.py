#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import copy
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms, models

try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =========================
# Data
# =========================
def build_dataloaders(data_dir: str, batch_size: int, val_split: float, num_workers: int):
    data_dir = Path(data_dir)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if train_dir.exists() and val_dir.exists():
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
        val_ds   = datasets.ImageFolder(str(val_dir),   transform=eval_tfms)
        class_names = train_ds.classes
    else:
        full_ds = datasets.ImageFolder(str(data_dir), transform=None)
        class_names = full_ds.classes
        n_total = len(full_ds)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(42)
        train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)
        train_ds = copy.deepcopy(full_ds)
        val_ds   = copy.deepcopy(full_ds)
        train_ds.transform = train_tfms
        val_ds.transform   = eval_tfms
        train_ds = Subset(train_ds, train_idx.indices if hasattr(train_idx, 'indices') else train_idx)
        val_ds   = Subset(val_ds,   val_idx.indices   if hasattr(val_idx, 'indices')   else val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, class_names


# =========================
# Model
# =========================
def build_model(num_classes: int):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def freeze_backbone(model: nn.Module):
    # conv1〜layer4 までを凍結、fc は学習
    set_requires_grad(model, False)
    set_requires_grad(model.fc, True)


def unfreeze(model: nn.Module, mode: str = "layer4"):
    """
    mode:
      - "layer4" (デフォルト): 最終ブロックのみ解凍
      - "all": 全層解凍
    """
    if mode == "all":
        set_requires_grad(model, True)
    elif mode == "layer4":
        # fc + layer4 を学習、他は凍結
        freeze_backbone(model)         # いったん全凍結（fc だけ True）
        set_requires_grad(model.layer4, True)
    else:
        raise ValueError(f"unsupported unfreeze mode: {mode}")


def param_groups_for_finetune(model: nn.Module, head_lr: float, backbone_lr: float, mode: str) -> List[dict]:
    """
    学習率を層ごとに分ける:
      - 全結合(fc): head_lr
      - 解凍されたバックボーン: backbone_lr
    """
    groups = []
    # fc は常に学習
    groups.append({"params": [p for p in model.fc.parameters() if p.requires_grad], "lr": head_lr})

    if mode == "all":
        groups.append({"params": [p for n,p in model.named_parameters()
                                  if p.requires_grad and not n.startswith("fc.")], "lr": backbone_lr})
    elif mode == "layer4":
        groups.append({"params": [p for p in model.layer4.parameters() if p.requires_grad], "lr": backbone_lr})
    return groups


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss, running_corrects, n = 0.0, 0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        n += inputs.size(0)
    return running_loss / n, running_corrects / n


@torch.no_grad()
def evaluate(model, loader, device, criterion, class_names):
    model.eval()
    running_loss, running_corrects, n = 0.0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        n += inputs.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    val_loss = running_loss / n
    val_acc  = running_corrects / n

    report, cm = None, None
    if HAS_SKLEARN:
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        cm = confusion_matrix(y_true, y_pred)
    return val_loss, val_acc, report, cm


# =========================
# Save / Export
# =========================
def save_class_map(class_names, path: str):
    idx2label = {int(i): name for i, name in enumerate(class_names)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx2label, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Class map saved to: {path}")


def export_onnx(model: nn.Module, onnx_path: str):
    was_training = model.training
    model.eval()

    device = torch.device('cpu')
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)
    model_cpu = copy.deepcopy(model).to("cpu")
    onnx_path_cpu = f'{onnx_path}.cpu32.onnx'
    torch.onnx.export(
        model_cpu, dummy, onnx_path_cpu,
        export_params=True, opset_version=12, do_constant_folding=True,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    if was_training:
        model.train()
    print(f"[INFO] ONNX model exported to: {onnx_path_cpu}")

    device = torch.device('cuda')
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)
    model_cuda = copy.deepcopy(model).to("cuda")
    onnx_path_cuda = f'{onnx_path}.cuda.onnx'
    torch.onnx.export(
        model_cuda, dummy, onnx_path_cuda,
        export_params=True, opset_version=12, do_constant_folding=True,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    if was_training:
        model.train()
    print(f"[INFO] ONNX model exported to: {onnx_path_cuda}")

    
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Transfer Learning (ResNet50, staged fine-tuning)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20, help="総エポック数 (stage1 + stage2)")
    parser.add_argument("--stage1_epochs", type=int, default=5, help="stage1: fc のみ学習するエポック数")
    parser.add_argument("--unfreeze", type=str, default="layer4", choices=["layer4", "all"],
                        help="stage2 で解凍する範囲")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--head_lr", type=float, default=1e-3, help="fc 用学習率")
    parser.add_argument("--backbone_lr", type=float, default=1e-4, help="解凍したバックボーン用学習率")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    parser.add_argument("--save_base_path", type=str, default="params/best_model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # Data
    train_loader, val_loader, class_names = build_dataloaders(
        args.data_dir, args.batch_size, args.val_split, args.num_workers
    )
    if not class_names:
        if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'classes'):
            class_names = train_loader.dataset.dataset.classes
        else:
            class_names = sorted([d.name for d in Path(args.data_dir).glob("*") if d.is_dir()])
    num_classes = len(class_names)
    print(f"[INFO] classes ({num_classes}): {class_names}")
    save_class_map(class_names, f'{args.save_base_path}.class_map.json')

    # Model
    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # =========
    # Stage 1: fc のみ学習（バックボーン凍結）
    # =========
    freeze_backbone(model)
    optim1 = optim.AdamW([p for p in model.fc.parameters() if p.requires_grad],
                         lr=args.head_lr, weight_decay=args.weight_decay)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=max(1, args.stage1_epochs))

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    total_epochs = args.epochs
    stage1_epochs = min(args.stage1_epochs, total_epochs)
    stage2_epochs = max(0, total_epochs - stage1_epochs)

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        print(f"\n===== Stage1 (fc only) Epoch {epoch}/{stage1_epochs} =====")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optim1)
        va_loss, va_acc, report, cm = evaluate(model, val_loader, device, criterion, class_names)
        sched1.step()
        dt = time.time() - t0
        print(f"[Train] loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"[Valid] loss: {va_loss:.4f}  acc: {va_acc:.4f}  (elapsed {dt:.1f}s)")
        if HAS_SKLEARN and report:
            print("\n[Valid] Classification Report:\n" + report)
            print("[Valid] Confusion Matrix:\n", cm)
        if va_acc > best_acc:
            best_acc, best_wts = va_acc, copy.deepcopy(model.state_dict())
            torch.save({"model_state": best_wts, "classes": class_names,
                        "val_acc": best_acc, "epoch": epoch, "arch": "resnet50"},
                       f'{args.save_base_path}.pth')
            print(f"[INFO] Best model updated (stage1) -> {args.save_base_path}.pth (acc={best_acc:.4f})")

    # 早期終了（全体 epochs が stage1 で尽きた）
    if stage2_epochs == 0:
        model.load_state_dict(best_wts)
        print(f"\n[RESULT] Best validation accuracy: {best_acc:.4f}")
        export_onnx(model, args.save_base_path)
        print("[DONE]")
        return

    # =========
    # Stage 2: バックボーン部分または全層を解凍して微調整
    # =========
    unfreeze(model, args.unfreeze)
    # パラメータグループ: fc は head_lr、解凍したバックボーンは backbone_lr
    param_groups = param_groups_for_finetune(model, head_lr=args.head_lr,
                                             backbone_lr=args.backbone_lr, mode=args.unfreeze)
    optim2 = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=max(1, stage2_epochs))

    for e in range(1, stage2_epochs + 1):
        t0 = time.time()
        global_epoch = stage1_epochs + e
        print(f"\n===== Stage2 (fine-tune {args.unfreeze}) Epoch {e}/{stage2_epochs}  (Total {global_epoch}/{total_epochs}) =====")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optim2)
        va_loss, va_acc, report, cm = evaluate(model, val_loader, device, criterion, class_names)
        sched2.step()
        dt = time.time() - t0
        print(f"[Train] loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"[Valid] loss: {va_loss:.4f}  acc: {va_acc:.4f}  (elapsed {dt:.1f}s)")
        if HAS_SKLEARN and report:
            print("\n[Valid] Classification Report:\n" + report)
            print("[Valid] Confusion Matrix:\n", cm)
        if va_acc > best_acc:
            best_acc, best_wts = va_acc, copy.deepcopy(model.state_dict())
            torch.save({"model_state": best_wts, "classes": class_names,
                        "val_acc": best_acc, "epoch": global_epoch, "arch": "resnet50"},
                       f'{args.save_base_path}.pth')
            print(f"[INFO] Best model updated (stage2) -> {args.save_base_path}.pth (acc={best_acc:.4f})")

    model.load_state_dict(best_wts)
    print(f"\n[RESULT] Best validation accuracy: {best_acc:.4f}")

    export_onnx(model, args.save_base_path)

    print("[DONE]")


if __name__ == "__main__":
    main()
