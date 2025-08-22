from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pathlib import Path
from uuid import uuid4
import time
from PIL import Image, UnidentifiedImageError
from utils_image_union import ImageUnionConverter
from estimator_with_unknown import Estimator

'''
サーバ起動：

uvicorn server_with_unknown:app --reload --port 3000 --host (host)
'''

'''
テストする場合：

curl.exe -X POST http://(host):3000/api/upload -F "file=@./training_images/Acinonyx jubatus/photo_381806451.jpeg" ./training_images/Acinonyx jubatus/photo_381806451.jpeg
'''


# 保存先ディレクトリ
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Image Receiver")
conv = ImageUnionConverter(allow_tempfile=False, convert_to_rgb=True)

est = Estimator(
    meta_src="./params/best_model.meta.json",
    label_src="./params/best_model.labels.json",
    weights="./params/best_model.pth",
    device="cuda"                 # GPUが無ければ "cpu"
)

# 開発中はCORSゆるめ（本番はoriginを絞る）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 例: ["http://localhost:19006", "https://your.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# React Native(Expo)からの画像＋位置情報 受信
@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    userId: Optional[str] = Form(None),
):
    # MIMEチェック（必要に応じて拡張）
    allowed = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    # 一時読み込み（サイズ制限はリバプロやWAFで併用推奨）
    data = await file.read()

    # 画像として開けるか簡易バリデーション
    try:
        print("filename=", file.filename, "ctype=", file.content_type, "size=", len(data))
        print("head bytes=", data[:16])  # JPEG: ff d8 ff..., PNG: 89 50 4E 47..., WebP: 'RIFF....WEBP'
        img = conv.to_union(data, as_type="pil")  # -> Image.Image（ディスク不使用）
        # 単一画像
        topk = 5
        result = est.predict_image(img, topk=topk)
        print("hoge")
        print(result["pred_label"], result["pmax"], result["energy"])
        print(result["topk"])  # 上位候補一覧
        pass
    except Exception:
        # Pillowの詳細を漏らさず汎用メッセージ
        raise HTTPException(status_code=400, detail="Invalid image file")

    return {
        "ok": True,
        "file": {
            "size": len(data),
            "mimetype": file.content_type,
        },
        "predict": {
            "best_label": result["pred_label"], 
            "pmax": result["pmax"], 
            "energy": result["energy"],
            "topk": result["topk"], 
        },
        "meta": {
            "userId": userId,
        },
    }
