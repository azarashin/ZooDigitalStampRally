# utils_image_union.py
from __future__ import annotations
from typing import Union, Literal, Optional
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import io
import tempfile

# 既存互換: 必要に応じて Path/str も返せるが、既定はメモリで完結
UnionImage = Union[str, Path, Image.Image, io.BytesIO]


class ImageUnionConverter:
    """
    bytes -> (PIL.Image / BytesIO / 一時ファイル Path/str)
    ※ 画像形式は jpg, png, webp に限定。
    ※ 既定では一時ファイルを作成しない（allow_tempfile=False）。
    """

    def __init__(self, *, allow_tempfile: bool = False, convert_to_rgb: bool = False) -> None:
        self.allow_tempfile = allow_tempfile
        self.convert_to_rgb = convert_to_rgb

    def to_union(
        self,
        data: bytes,
        as_type: Literal["pil", "bytesio", "path", "str"] = "pil",
        *,
        suffix: Optional[str] = None
    ) -> UnionImage:
        """
        data(bytes) を指定の型に変換:
          - "pil"     -> PIL.Image.Image（メモリのみ）
          - "bytesio" -> io.BytesIO（メモリのみ）
          - "path"    -> 一時ファイルの Path（allow_tempfile=True のときのみ）
          - "str"     -> 一時ファイルのパス文字列（同上）
        suffix: ".jpg" 等の拡張子（"path"/"str" のときのみ有効）
        """
        if not data:
            raise ValueError("empty image data")

        fmt = self._detect_format(data)   # "jpeg" | "png" | "webp"（想定外なら ValueError）

        if as_type == "pil":
            img = Image.open(io.BytesIO(data))
            img.load()  # 破損検知
            if self.convert_to_rgb and img.mode != "RGB":
                img = img.convert("RGB")
            return img

        if as_type == "bytesio":
            # そのまま渡したいライブラリがある場合に便利
            return io.BytesIO(data)

        # 以下はファイルパスを必要とする外部ライブラリ向け（極力使わない）
        if not self.allow_tempfile:
            raise RuntimeError('Temporary file is disabled. Use "pil" or "bytesio" instead.')

        ext = suffix or { "jpeg": ".jpg", "png": ".png", "webp": ".webp" }[fmt]
        p = self._bytes_to_tempfile(data, ext)
        return p if as_type == "path" else str(p)

    # ----------------- helpers -----------------

    @staticmethod
    def _detect_format(data: bytes) -> str:
        """
        マジックナンバーで jpg/png/webp を判定。該当しなければ ValueError。
        """
        head = data[:12]
        if head.startswith(b"\xFF\xD8\xFF"):
            return "jpeg"
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return "webp"
        raise ValueError("unsupported image format (expected jpg/png/webp)")

    @staticmethod
    def _bytes_to_tempfile(data: bytes, suffix: str) -> Path:
        f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            f.write(data)
            f.flush()
            return Path(f.name)
        finally:
            f.close()
