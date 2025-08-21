# transfer_open_set_single_dir_export.py
import os, json, argparse, random
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices, transform):
        self.base = base; self.indices = indices; self.transform = transform
        self.classes = base.classes; self.class_to_idx = base.class_to_idx
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        path, y = self.base.samples[idx]
        img = self.base.loader(path)
        img = self.transform(img)
        return img, y

# -------------------- Utils --------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------- Dataset & Split --------------------
def build_dataset(root: str, img_size: int) -> torchvision.datasets.ImageFolder:
    mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
    base_tf = T.Compose([
        T.Resize(int(img_size*1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean,std),
    ])
    return torchvision.datasets.ImageFolder(root=root, transform=base_tf)

def stratified_split_indices(ds: torchvision.datasets.ImageFolder, val_ratio: float, seed: int,
                             split_cache: str = None):
    if split_cache and os.path.exists(split_cache):
        with open(split_cache, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("num_samples") == len(ds.samples):
            return cache["train_idx"], cache["val_idx"]

    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, (_, y) in enumerate(ds.samples):
        by_class.setdefault(y, []).append(idx)

    train_idx, val_idx = [], []
    for _, idxs in by_class.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        val_idx.extend(idxs[:n_val]); train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx); rng.shuffle(val_idx)

    if split_cache:
        ensure_dir(os.path.dirname(split_cache))
        with open(split_cache, "w", encoding="utf-8") as f:
            json.dump({"num_samples": len(ds.samples),
                       "train_idx": train_idx, "val_idx": val_idx}, f)
        print(f"[split] saved to {split_cache} (train={len(train_idx)}, val={len(val_idx)})")

    return train_idx, val_idx

def build_loaders_single_dir(root: str, img_size: int, batch_size: int, workers: int,
                             val_ratio: float, seed: int, split_cache: str = None):
    ds = build_dataset(root, img_size)
    classes = ds.classes
    train_idx, val_idx = stratified_split_indices(ds, val_ratio, seed, split_cache)

    mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
    tf_train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.ToTensor(), T.Normalize(mean,std),
    ])
    tf_val = T.Compose([
        T.Resize(int(img_size*1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize(mean,std),
    ])


    tr_ds = TransformSubset(ds, train_idx, tf_train)
    va_ds = TransformSubset(ds, val_idx, tf_val)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return tr_loader, va_loader, classes

# -------------------- Model --------------------
class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # conv~layer4
        self.feat_dim = 2048
        self.head = nn.Linear(self.feat_dim, num_classes)
    def forward(self, x):
        f = self.backbone(x)
        f = F.adaptive_avg_pool2d(f,1).flatten(1)
        logits = self.head(f)
        return logits

# -------------------- Train --------------------
def train_linear(model, loader, epochs=3, lr=1e-2, wd=1e-4):
    for p in model.backbone.parameters(): p.requires_grad = False
    opt = torch.optim.SGD(model.head.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    ce = nn.CrossEntropyLoss()
    model.to(DEVICE)
    for ep in range(1, epochs+1):
        model.train(); tot=0; n=0
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y); loss.backward(); opt.step()
            tot += loss.item()*y.size(0); n += y.size(0)
        print(f"[linear] {ep}/{epochs} loss={tot/n:.4f}")

def finetune_last_block(model, loader, epochs=5, lr=1e-3, wd=1e-4):
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in list(model.backbone.children())[-1].parameters():  # layer4
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    ce = nn.CrossEntropyLoss()
    model.to(DEVICE)
    for ep in range(1, epochs+1):
        model.train(); tot=0; n=0
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y); loss.backward(); opt.step()
            tot += loss.item()*y.size(0); n += y.size(0)
        print(f"[ft] {ep}/{epochs} loss={tot/n:.4f}")

# -------------------- OOD Thresholds --------------------
@torch.no_grad()
def collect_logits(model, loader, T=1.0):
    model.eval(); en_list=[]; pmax_list=[]
    for x,_ in loader:
        x = x.to(DEVICE)
        logits = model(x)
        en = -T * torch.logsumexp(logits/T, dim=1)   # 低いほどID
        pmax = logits.softmax(1).max(1).values       # 高いほどID
        en_list.append(en.cpu()); pmax_list.append(pmax.cpu())
    return torch.cat(en_list), torch.cat(pmax_list)

def calibrate_thresholds(model, val_loader, q=0.05, T=2.0):
    en, pmax = collect_logits(model, val_loader, T=T)
    # Energy: 低いほどID → 受け入れたい割合(1-q)まで許容するには 1-q 分位を上限に
    tau_energy = torch.quantile(en, 1.0 - q).item()   # ★ 変更: 1.0 - q

    # MSP: 高いほどID → 低い方の q 分位を下限に
    tau_msp    = torch.quantile(pmax, q).item()       # ★ 変更: q

    return tau_energy, tau_msp, T
# -------------------- Save / Export --------------------
def save_state_dict(model: nn.Module, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    print(f"[save] state_dict -> {path}")

def export_onnx(model: nn.Module, img_size: int, path: str, dynamic: bool=True, opset: int=13):
    ensure_dir(os.path.dirname(path))
    orig_device = next(model.parameters()).device  # ★ 元デバイスを保存
    model = model.to("cpu").eval()
    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    input_names = ["input"]; output_names = ["logits"]
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic else None
    torch.onnx.export(
        model, dummy, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True
    )
    model.to(orig_device)  # ★ 元デバイスに戻す
    print(f"[onnx] exported -> {path}")

def save_meta_json(classes: List[str], tau_energy: float, tau_msp: float, T: float, img_size: int, path: str):
    ensure_dir(os.path.dirname(path))
    meta = {
        "classes": classes,
        "thresholds": {"tau_energy": tau_energy, "tau_msp": tau_msp, "temperature": T},
        "preprocess": {
            "resize": int(img_size*1.14), "center_crop": img_size,
            "mean": [0.485,0.456,0.406], "std": [0.229,0.224,0.225]
        },
        "model_input": {"shape": [None, 3, img_size, img_size], "dtype": "float32"},
        "outputs": ["logits"],
        "decision_rule": "accept if (energy <= tau_energy) AND (max_softmax >= tau_msp), else -1"
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[meta] saved -> {path}")

def save_labels_json(classes: list, path: str):
    """idx -> class_name の辞書を JSON で保存"""
    ensure_dir(os.path.dirname(path))
    idx_to_class = {str(i): name for i, name in enumerate(classes)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2, ensure_ascii=False)
    print(f"[labels] saved -> {path}")
    
# -------------------- Inference helper (optional) --------------------
@torch.no_grad()
def predict_with_thresholds(model, x, tau_energy, tau_msp, T=2.0):
    model.eval()
    logits = model(x.to(DEVICE))
    p = logits.softmax(1)
    pmax, c = p.max(1)
    energy = -T * torch.logsumexp(logits/T, dim=1)
    accept = (energy <= tau_energy) & (pmax >= tau_msp)
    pred = torch.where(accept, c, torch.full_like(c, -1))
    return pred, pmax, energy

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Transfer learning + open-set (single folder) with .pth & dual ONNX export")
    ap.add_argument("--data_root", type=str, required=True, help="e.g. ./training_images")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_cache", type=str, default="./params/splits.json")

    # training hparams
    ap.add_argument("--linear_epochs", type=int, default=10)
    ap.add_argument("--ft_epochs", type=int, default=15)
    ap.add_argument("--linear_lr", type=float, default=1e-2)
    ap.add_argument("--ft_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)

    # OOD threshold
    ap.add_argument("--q", type=float, default=0.05, help="quantile for thresholds")
    ap.add_argument("--T", type=float, default=2.0, help="temperature for energy")

    # outputs
    ap.add_argument("--save_dir", type=str, default="./params")
    ap.add_argument("--pth_name", type=str, default=None)
    ap.add_argument("--onnx_cpu_name", type=str, default=None)
    ap.add_argument("--onnx_cuda_name", type=str, default=None)
    ap.add_argument("--labels_name", type=str, default=None)
    ap.add_argument("--meta_name", type=str, default=None)

    # unified naming
    ap.add_argument("--file_name", type=str, default='best_model',
                    help="指定時: {XXX}.pth, {XXX}.cpu.onnx, {XXX}.cuda.onnx, {XXX}.meta.json を出力")
    args = ap.parse_args()

    set_seed(args.seed)

    # ===== name resolution =====
    if args.file_name:  # ここがご要望の挙動
        base = args.file_name
        pth_name = f"{base}.pth"
        onnx_cpu_name = f"{base}.cpu.onnx"
        onnx_cuda_name = f"{base}.cuda.onnx"
        class_name = f"{base}.class.json"
        meta_name = f"{base}.meta.json"
        labels_name    = f"{base}.labels.json"
    else:
        pth_name       = args.pth_name
        onnx_cpu_name  = args.onnx_cpu_name
        onnx_cuda_name = args.onnx_cuda_name
        meta_name      = args.meta_name
        labels_name    = args.labels_name

    # ===== data/loaders =====
    tr_loader, va_loader, classes = build_loaders_single_dir(
        root=args.data_root, img_size=args.img_size, batch_size=args.batch_size, workers=args.workers,
        val_ratio=args.val_ratio, seed=args.seed, split_cache=args.split_cache
    )
    print("Classes:", classes)

    # ===== model & train =====
    model = Model(num_classes=len(classes)).to(DEVICE)
    train_linear(model, tr_loader, epochs=args.linear_epochs, lr=args.linear_lr, wd=args.wd)
    finetune_last_block(model, tr_loader, epochs=args.ft_epochs, lr=args.ft_lr, wd=args.wd)

    # ===== thresholds =====
    tau_energy, tau_msp, T = calibrate_thresholds(model, va_loader, q=args.q, T=args.T)
    print(f"tau_energy={tau_energy:.4f}, tau_msp={tau_msp:.4f}, T={T}")

    # ===== save / export =====
    ensure_dir(args.save_dir)
    save_state_dict(model, os.path.join(args.save_dir, pth_name))

    # ONNX: CPU 用（ファイル名で区別）
    export_onnx(model, args.img_size, os.path.join(args.save_dir, onnx_cpu_name), dynamic=True, opset=13)
    # ONNX: CUDA 用（同一グラフだが運用上の区別で別ファイルに保存）
    export_onnx(model, args.img_size, os.path.join(args.save_dir, onnx_cuda_name), dynamic=True, opset=13)

    save_labels_json(classes, os.path.join(args.save_dir, labels_name))

    save_meta_json(classes, tau_energy, tau_msp, T, args.img_size, os.path.join(args.save_dir, meta_name))

    # 任意のワンショット検証
    for x,_ in va_loader:
        pred, pmax, energy = predict_with_thresholds(model, x, tau_energy, tau_msp, T)
        print("pred (first 10):", pred[:10].tolist())
        break

if __name__ == "__main__":
    main()
