# infer_animals_pth.py
import os, json, argparse, glob
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import csv

# ---------- Model (must match training) ----------
class Model(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # conv..layer4
        self.feat_dim = 2048
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        logits = self.head(f)
        return logits

class Estimator:

    # ---------- Preprocess from meta.json ----------
    def build_transform(self, meta: dict) -> T.Compose:
        pp = meta["preprocess"]
        resize = pp["resize"]
        crop = pp["center_crop"]
        mean = pp["mean"]; std = pp["std"]
        return T.Compose([
            T.Resize(resize),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    # ---------- I/O helpers ----------
    def load_meta(self, meta_path: str) -> dict:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_labels(self, labels_path: str) -> dict:
        with open(labels_path, "r", encoding="utf-8") as f:
            return json.load(f)  # {"0":"lion",...}

    # ---------- Predict with thresholds ----------
    @torch.no_grad()
    def predict_batch(self, model: nn.Module, x: torch.Tensor, tau_energy: float, tau_msp: float, T: float):
        model.eval()
        dev = next(model.parameters()).device
        x = x.to(dev)
        logits = model(x)                                # (B, C)
        p = logits.softmax(1)                            # prob
        pmax, c = p.max(1)                               # MSP & argmax
        energy = -T * torch.logsumexp(logits / T, dim=1) # Energy（低いほどID）
        accept = (energy <= tau_energy) & (pmax >= tau_msp)
        pred = torch.where(accept, c, torch.full_like(c, -1))
        return pred.cpu(), pmax.cpu(), energy.cpu(), p.cpu()

    def __init__(self, meta_src, label_src, weights, device, batch_size = 16):
        # Load meta / labels
        meta = self.load_meta(meta_src)
        self._labels_map = self.load_labels(label_src)           # {"0":"lion",...}
        classes = meta.get("classes", [self._labels_map[str(i)] for i in range(len(self._labels_map))])
        self._num_classes = len(classes)
        self._tau_energy = meta["thresholds"]["tau_energy"]
        self._tau_msp    = meta["thresholds"]["tau_msp"]
        self._T          = meta["thresholds"]["temperature"]
        self._batch_size = batch_size

        # Build model & load weights
        self._model = Model(num_classes=self._num_classes).to(device)
        sd = torch.load(weights, map_location=device)
        self._model.load_state_dict(sd, strict=True)

        # Preprocess
        self._tf = self.build_transform(meta)

    def predict_paths(self, paths, topk = 10):
        # Batched inference
        rows = []
        batch_imgs, batch_paths = [], []
        for pth in paths:
            try:
                img = Image.open(pth).convert("RGB")
            except Exception as e:
                print(f"[warn] failed to open {pth}: {e}")
                continue
            x = self._tf(img)  # (3,H,W)
            batch_imgs.append(x)
            batch_paths.append(pth)

            if len(batch_imgs) == self._batch_size:
                X = torch.stack(batch_imgs, 0)  # (B,3,H,W)
                pred, pmax, energy, prob = self.predict_batch(self._model, X, self._tau_energy, self._tau_msp, self._T)
                for i in range(X.size(0)):
                    idx = int(pred[i].item())
                    label = self._labels_map[str(idx)] if idx >= 0 else "-1"
                    # top-k (only if ID)
                    if idx >= 0:
                        modified_topk = min(topk, self._num_classes)
                        topk_idx = torch.topk(prob[i], k=modified_topk).indices.tolist()
                        topk_pairs = [(j, float(prob[i][j])) for j in topk_idx]
                    else:
                        topk_pairs = []
                    rows.append({
                        "path": batch_paths[i],
                        "pred_idx": idx,
                        "pred_label": label,
                        "pmax": float(pmax[i]),
                        "energy": float(energy[i]),
                        "topk": topk_pairs
                    })
                batch_imgs, batch_paths = [], []

        # last partial batch
        if batch_imgs:
            X = torch.stack(batch_imgs, 0)
            pred, pmax, energy, prob = self.predict_batch(self._model, X, self._tau_energy, self._tau_msp, self._T)
            for i in range(X.size(0)):
                idx = int(pred[i].item())
                label = self._labels_map[str(idx)] if idx >= 0 else "-1"
                if idx >= 0:
                    modified_topk = min(topk, self._num_classes)
                    topk_idx = torch.topk(prob[i], k=modified_topk).indices.tolist()
                    topk_pairs = [(j, float(prob[i][j])) for j in topk_idx]
                else:
                    topk_pairs = []
                rows.append({
                    "path": batch_paths[i],
                    "pred_idx": idx,
                    "pred_label": label,
                    "pmax": float(pmax[i]),
                    "energy": float(energy[i]),
                    "topk": topk_pairs
                })
        return rows
    
    def predict_image_path(self, image_path, topk = 10):
        # Batched inference
        rows = []
        batch_imgs, batch_paths = [], []
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[warn] failed to open {image_path}: {e}")
            return None
        x = self._tf(img)  # (3,H,W)
        batch_imgs.append(x)
        batch_paths.append(image_path)

        X = torch.stack(batch_imgs, 0)  # (B,3,H,W)
        pred, pmax, energy, prob = self.predict_batch(self._model, X, self._tau_energy, self._tau_msp, self._T)
        for i in range(X.size(0)):
            idx = int(pred[i].item())
            label = self._labels_map[str(idx)] if idx >= 0 else "-1"
            # top-k (only if ID)
            if idx >= 0:
                modified_topk = min(topk, self._num_classes)
                topk_idx = torch.topk(prob[i], k=modified_topk).indices.tolist()
                topk_pairs = [(j, float(prob[i][j])) for j in topk_idx]
            else:
                topk_pairs = []
            return {
                "path": batch_paths[i],
                "pred_idx": idx,
                "pred_label": label,
                "pmax": float(pmax[i]),
                "energy": float(energy[i]),
                "topk": topk_pairs
            }

        return None

    def predict_image(self, image, topk = 10):
        # Batched inference
        rows = []
        batch_imgs, batch_paths = [], []

        x = self._tf(image)  # (3,H,W)
        batch_imgs.append(x)

        X = torch.stack(batch_imgs, 0)  # (B,3,H,W)
        pred, pmax, energy, prob = self.predict_batch(self._model, X, self._tau_energy, self._tau_msp, self._T)
        for i in range(X.size(0)):
            idx = int(pred[i].item())
            label = self._labels_map[str(idx)] if idx >= 0 else "-1"
            # top-k (only if ID)
            if idx >= 0:
                modified_topk = min(topk, self._num_classes)
                topk_idx = torch.topk(prob[i], k=modified_topk).indices.tolist()
                topk_pairs = [(j, float(prob[i][j])) for j in topk_idx]
            else:
                topk_pairs = []
            return {
                "path": None,
                "pred_idx": idx,
                "pred_label": label,
                "pmax": float(pmax[i]),
                "energy": float(energy[i]),
                "topk": topk_pairs
            }

        return None

def list_images(input_path: str) -> List[str]:
    # file / dir / glob に対応
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(input_path, e)))
        return sorted(files)
    # glob pattern
    return sorted(glob.glob(input_path))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inference with .pth (transfer learning + open-set threshold)")
    ap.add_argument("--weights", required=True, help="path to .pth (state_dict)")
    ap.add_argument("--meta", required=True, help="path to meta.json (contains thresholds & preprocess)")
    ap.add_argument("--labels", required=True, help="path to labels.json (idx->class)")
    ap.add_argument("--input", required=True, help="image file | directory | glob pattern")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--save_csv", type=str, default=None, help="optional: save results to CSV")
    ap.add_argument("--topk", type=int, default=3, help="also print top-k probabilities (within ID only)")
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    est = Estimator(args.meta, args.labels, args.weights, device, args.batch_size)


    # Images
    paths = list_images(args.input)
    if not paths:
        print(f"No images found for: {args.input}")
        exit()

    if len(paths) == 1:
        rows = [est.predict_image_path(paths[0], args.topk)]
    else:
        rows = est.predict_paths(paths, args.topk)

    # Print summary
    for r in rows:
        if r["pred_idx"] >= 0:
            print(f'{r["path"]} -> {r["pred_label"]} (idx={r["pred_idx"]}) '
                f'pmax={r["pmax"]:.3f} energy={r["energy"]:.3f} topk={r["topk"]}')
        else:
            print(f'{r["path"]} -> -1 (該当なし) pmax={r["pmax"]:.3f} energy={r["energy"]:.3f}')

    # Save CSV (optional)
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "pred_idx", "pred_label", "pmax", "energy", "topk"])
            for r in rows:
                writer.writerow([r["path"], r["pred_idx"], r["pred_label"], r["pmax"], r["energy"], r["topk"]])
