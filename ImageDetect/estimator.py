#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional, Union
from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class ResNet50Classifier:
    """
    学習スクリプトで保存した best_model.pth（dict: model_state/classes/arch…）を読み込み、
    画像1枚 or 複数枚の分類推論を提供するクラス。
    - 非正方形画像: 中央の最大正方形を抽出 -> 224x224 -> 正規化 -> 推論
    """

    def __init__(self,
                 model_pth: Union[str, Path],
                 class_map: Optional[Union[str, Path]] = None,
                 device: str = "cuda"):
        """
        Args:
            model_pth: 学習済みモデルの .pth パス（学習スクリプト出力）
            class_map: class_map.json（index -> label）。省略可（ckpt['classes'] があればそれを使用）
            device: "cuda" or "cpu"
        """
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

        # --- チェックポイント読込 ---
        ckpt = torch.load(str(model_pth), map_location="cpu")
        if "model_state" not in ckpt:
            raise ValueError("Invalid checkpoint: 'model_state' not found.")

        self.arch = str(ckpt.get("arch", "resnet50")).lower()

        # --- クラス名の決定（class_map.json > ckpt['classes'] > fc.weight 形状から推定） ---
        self.class_names = self._load_class_names(ckpt, class_map)
        if not self.class_names:
            state = ckpt["model_state"]
            if "fc.weight" in state:
                num_classes = int(state["fc.weight"].shape[0])
                self.class_names = [str(i) for i in range(num_classes)]
            else:
                raise ValueError("Cannot determine number of classes. Provide class_map.json or classes in ckpt.")

        self.num_classes = len(self.class_names)

        # --- モデル構築＆重み反映 ---
        self.model = self._build_model(self.num_classes, self.arch)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval().to(self.device)

        # --- 前処理（正規化は ImageNet 準拠） ---
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # ===== Public APIs =====================================================

    @torch.inference_mode()
    def predict(self,
                img: Union[str, Path, Image.Image],
                topk: int = 5) -> dict:
        """
        単一画像の推論。

        Returns:
            {
              "pred_index": int,
              "pred_label": str,
              "pred_confidence": float,
              "topk": List[Tuple[int, str, float]]  # (index, label, prob)
            }
        """
        pil = self._ensure_image(img)
        pil_sq = self._center_max_square(pil)

        x = self.transform(pil_sq).unsqueeze(0).to(self.device)  # [1,3,224,224]
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)          # [C]

        k = min(topk, probs.numel())
        conf, idx = torch.topk(probs, k=k)
        idx = idx.tolist()
        conf = conf.tolist()

        topk_list = [(int(i), self.class_names[i], float(c)) for i, c in zip(idx, conf)]
        best_idx, best_label, best_conf = topk_list[0]
        return {
            "pred_index": best_idx,
            "pred_label": best_label,
            "pred_confidence": best_conf,
            "topk": topk_list,
        }

    @torch.inference_mode()
    def predict_many(self,
                     imgs: List[Union[str, Path, Image.Image]],
                     topk: int = 5) -> List[dict]:
        """
        複数画像の推論（1枚ずつ処理）。必要に応じてバッチ化に差し替え可。
        """
        results = []
        for img in imgs:
            results.append(self.predict(img, topk=topk))
        return results

    def get_class_names(self) -> List[str]:
        """クラス名の配列（index -> name）を返す。"""
        return list(self.class_names)

    # ===== Internals =======================================================

    def _build_model(self, num_classes: int, arch: str) -> nn.Module:
        if arch != "resnet50":
            raise ValueError(f"Unsupported arch: {arch} (expected 'resnet50')")
        model = models.resnet50(weights=None)  # 学習済み重みは ckpt から読み込むため不要
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def _load_class_names(self,
                          ckpt: dict,
                          class_map_path: Optional[Union[str, Path]]) -> List[str]:
        # 明示 JSON が最優先
        if class_map_path:
            with open(str(class_map_path), "r", encoding="utf-8") as f:
                idx2label = json.load(f)  # {0:"cat"} or {"0":"cat"}
            items = sorted(((int(k), v) for k, v in idx2label.items()), key=lambda x: x[0])
            return [v for _, v in items]
        # ckpt に classes があれば採用
        if "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
            return list(ckpt["classes"])
        return []

    @staticmethod
    def _ensure_image(img: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        p = Path(img)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return Image.open(p).convert("RGB")

    @staticmethod
    def _center_max_square(img: Image.Image) -> Image.Image:
        """非正方形画像の中央から最大の正方形を抽出。"""
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top  = (h - s) // 2
        return img.crop((left, top, left + s, top + s))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inference with .pth (ResNet50) + center square crop")
    ap.add_argument("--model_pth", type=str, required=True, help="学習済みモデル .pth（学習スクリプトの出力）")
    ap.add_argument("--input", type=str, required=True, help="画像パス or ディレクトリ")
    ap.add_argument("--class_map", type=str, default="", help="class_map.json（任意）")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    
    clf = ResNet50Classifier(
        model_pth=args.model_pth,
        class_map=args.class_map,   # 省略可（ckpt['classes'] があれば不要）
        device=args.device                 # GPUが無ければ "cpu"
    )

    # 単一画像
    result = clf.predict(args.input, topk=args.topk)
    print(result["pred_label"], result["pred_confidence"])
    print(result["topk"])  # 上位候補一覧

    # 複数画像
    batch_results = clf.predict_many([args.input, args.input])
    for result in batch_results:
        print(result["pred_label"], result["pred_confidence"])
        print(result["topk"])  # 上位候補一覧
