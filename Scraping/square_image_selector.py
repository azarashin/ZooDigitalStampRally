#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Square Crop Tool (overlay-safe)
- 画像へのオーバレイは表示用だけに描画し、保存画像には含めない
- 表示縮尺と元画像の座標を厳密対応させる
"""

import shutil
import os
import cv2
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WINDOW_NAME = "Square Crop Tool"
INSTR_LINES = [
    "Drag: draw square  |  Enter: save crop  |  DEL: delete file",
    "n/Space: next  |  r: reset selection  |  q/ESC: quit"
]

# Key sets
KEY_ENTER = {10, 13}
KEY_ESC = {27}
KEY_SPACE = {32}
KEY_DEL = {46, 127, 3014656}   # 46(DEL), 127(ASCII DEL), 3014656(OpenCV special)
KEY_N = {ord('n'), ord('N')}
KEY_R = {ord('r'), ord('R')}
KEY_Q = {ord('q'), ord('Q')}

# 表示サイズ上限（大きすぎる画像は縮小表示）
MAX_VIEW_W = 1600
MAX_VIEW_H = 1000

def fit_to_view(w, h):
    """表示用の縮小倍率と表示サイズを算出"""
    scale = min(MAX_VIEW_W / w, MAX_VIEW_H / h, 1.0)
    vw, vh = int(round(w * scale)), int(round(h * scale))
    return scale, vw, vh

class SquareSelector:
    """座標は常に『元画像スケール』で保持。表示は縮小されていてもOK。"""
    def __init__(self, base_shape: Tuple[int, int, int], scale: float):
        self.h, self.w = base_shape[:2]
        self.scale = scale
        self.start_o: Optional[Tuple[int, int]] = None  # original coords
        self.end_o: Optional[Tuple[int, int]] = None
        self.final_square_o: Optional[Tuple[int, int, int, int]] = None
        self.is_dragging = False

    def reset(self):
        self.start_o = None
        self.end_o = None
        self.final_square_o = None
        self.is_dragging = False

    def _clamp_o(self, x, y):
        return max(0, min(x, self.w - 1)), max(0, min(y, self.h - 1))

    def _v2o(self, xv, yv):
        """view座標→original座標"""
        xo = int(round(xv / self.scale))
        yo = int(round(yv / self.scale))
        return self._clamp_o(xo, yo)

    def _square_from_drag_o(self, p0, p1):
        """元画像座標で正方形を作る"""
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        sx = 1 if dx >= 0 else -1
        sy = 1 if dy >= 0 else -1
        side = max(abs(dx), abs(dy))
        x2 = x0 + sx * side
        y2 = y0 + sy * side
        x2 = max(0, min(x2, self.w - 1))
        y2 = max(0, min(y2, self.h - 1))
        side = min(abs(x2 - x0), abs(y2 - y0))
        x2 = x0 + sx * side
        y2 = y0 + sy * side
        x_min = min(x0, x2)
        y_min = min(y0, y2)
        return (x_min, y_min, side, side)

    def on_mouse(self, event, x_view, y_view, flags, param):
        # 受け取るのはview座標 → originalに変換
        xo, yo = self._v2o(x_view, y_view)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_o = (xo, yo)
            self.end_o = (xo, yo)
            self.is_dragging = True
            self.final_square_o = None
        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging and self.start_o:
            self.end_o = (xo, yo)
        elif event == cv2.EVENT_LBUTTONUP and self.is_dragging and self.start_o:
            self.end_o = (xo, yo)
            self.is_dragging = False
            sq = self._square_from_drag_o(self.start_o, self.end_o)
            self.final_square_o = sq if sq[2] > 0 else None

    def get_preview_square_o(self):
        if self.is_dragging and self.start_o and self.end_o:
            sq = self._square_from_drag_o(self.start_o, self.end_o)
            if sq[2] > 0:
                return sq
        return None

def draw_instructions(view_img):
    y = 24
    for line in INSTR_LINES:
        cv2.putText(view_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26

def draw_square_on_view(view_img, sq_o, scale, color=(0, 255, 0)):
    x, y, w, h = sq_o
    xv = int(round(x * scale))
    yv = int(round(y * scale))
    wv = int(round(w * scale))
    hv = int(round(h * scale))
    cv2.rectangle(view_img, (xv, yv), (xv + wv, yv + hv), color, 2)

def crop_from_base_and_save(base_img, img_path: Path, square_o) -> bool:
    x, y, w, h = square_o
    H, W = base_img.shape[:2]
    x2 = min(x + w, W)
    y2 = min(y + h, H)
    x = max(0, x); y = max(0, y)
    if x >= x2 or y >= y2:
        logging.error(f"選択範囲が不正: {square_o} @ {img_path}")
        return False
    crop = base_img[y:y2, x:x2].copy()
    ok = cv2.imwrite(str(img_path), crop)
    if ok:
        logging.info(f"保存: {img_path}  (size={crop.shape[1]}x{crop.shape[0]})")
    else:
        logging.error(f"保存失敗: {img_path}")
    return ok

def main():
    parser = argparse.ArgumentParser(description="正方形領域を切り出して再保存")
    parser.add_argument("input", help="入力画像ディレクトリパス")
    parser.add_argument("-o", "--output", default="../ImageDetect/training_images", help="出力ディレクトリ（学習データを格納）")
    args = parser.parse_args()

    root = Path(args.input).expanduser().resolve()
    out_dir = args.output
    files = sorted(list(root.glob("*.jpg")) + list(root.glob("*.jpeg")))
    for file in files:
        print(file)
    if not files:
        logging.warning(f"JPGが見つかりません: {root}")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    i = 0
    while i < len(files):
        path = files[i]
        print(f'{i}, {path}')
        base_img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # ← 保存用の元画像（描き込まない）
        if base_img is None:
            logging.error(f"読み込み失敗: {path.name}")
            i += 1
            continue

        H, W = base_img.shape[:2]
        scale, VW, VH = fit_to_view(W, H)
        selector = SquareSelector(base_img.shape, scale)
        cv2.setMouseCallback(WINDOW_NAME, selector.on_mouse)

        while True:
            # 表示用のコピーを毎フレーム作成（上書き保存はしない）
            view_img = cv2.resize(base_img, (VW, VH), interpolation=cv2.INTER_AREA)
            key = cv2.waitKeyEx(16) & 0xFFFFFFFF  # 特殊キー対応
            if not key in KEY_ENTER:
                draw_instructions(view_img)

            # プレビュー
            sq_prev = selector.get_preview_square_o()
            if sq_prev:
                draw_square_on_view(view_img, sq_prev, scale, color=(0, 255, 255))
            if selector.final_square_o:
                draw_square_on_view(view_img, selector.final_square_o, scale, color=(0, 255, 0))

            title = f"{WINDOW_NAME}  -  {path.name}  ({i+1}/{len(files)})"
            cv2.imshow(WINDOW_NAME, view_img)
            
            out_path_parts = str(path).replace('\\', '/').split('/')
            out_sub_dir = f'{out_dir}/{out_path_parts[-2]}'
            out_path = f'{out_sub_dir}/{out_path_parts[-1]}'
            if not os.path.exists(out_sub_dir):
                os.mkdir(out_sub_dir)

            if key in KEY_Q or key in KEY_ESC:
                cv2.destroyAllWindows()
                return
            elif key in KEY_R:
                selector.reset()
                logging.info("選択リセット")
            elif key in KEY_SPACE or key in KEY_N:
                logging.info(f"そのまま採用: {i}, {path.name}")
                shutil.copy(path, out_path)
                i += 1
                break
            elif key in KEY_DEL:
                logging.info(f"学習データに含めない: {i}, {path.name}")
                i += 1
                break
            elif key in KEY_ENTER:
                if selector.final_square_o is None:
                    logging.error(f"正方形が選択されていません: {path.name}")
                else:
                    if crop_from_base_and_save(base_img, out_path, selector.final_square_o):
                        i += 1
                        break
                    else:
                        i += 1
                        break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
