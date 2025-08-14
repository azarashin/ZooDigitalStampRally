import argparse
import os
import cv2

# 引数で指定されたパスの動画を読み込み、中央部分の最大正方形領域を抽出して
# 256x256サイズに縮小し、ファイルに保存する

def center_square_crop(frame):
    """中央の最大正方形領域を切り出す。"""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0+side, x0:x0+side]

def process_video_frames(input_path, output_dir, size=256, img_format="png"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        crop = center_square_crop(frame)
        resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

        filename = f"frame_{frame_idx:06d}.{img_format}"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, resized)

        frame_idx += 1

    cap.release()
    return frame_idx

def main():
    parser = argparse.ArgumentParser(description="中央の最大正方形領域を切り出して256x256の画像として保存")
    parser.add_argument("input", help="入力動画パス")
    parser.add_argument("-o", "--output", default="frames", help="出力ディレクトリ")
    parser.add_argument("--size", type=int, default=256, help="出力画像サイズ（正方形）")
    parser.add_argument("--format", choices=["png", "jpg", "jpeg"], default="png", help="出力画像形式")
    args = parser.parse_args()

    total_frames = process_video_frames(args.input, args.output, size=args.size, img_format=args.format)
    print(f"完了: {total_frames} 枚の画像を {args.output} に保存しました")

if __name__ == "__main__":
    main()
