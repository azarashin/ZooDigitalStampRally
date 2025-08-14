#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iNaturalist から Panthera leo (taxon_id=42048) の CC0/CC-BY 画像を一括DL
- 画像: ./lion_images に保存（ファイル名: photo_{photo_id}.jpg）
- メタ情報: ./lion_images/metadata.csv に追記（重複は自動スキップ）
必要ライブラリ: requests
"""

import csv
import os
import sys
import time
import requests
from urllib.parse import urlencode

class INaturalistDownloader:

    def __init__(self, taxon_id, out_dir):
        self.TAXON_ID = taxon_id
        self.OUT_DIR = out_dir
        self.META_CSV = os.path.join(self.OUT_DIR, "metadata.csv")

        # 取得対象のライセンス（写真のライセンス）
        self.PHOTO_LICENSES = ["cc0", "cc-by"]  # iNat APIの指定値
        self.QUALITY_GRADE = "research"         # 研究グレード（精度高め）
        self.PER_PAGE = 200                     # 1ページの件数（最大200）
        self.MAX_PAGES = None                   # None なら全ページ。数値で上限指定可（例: 50）
        self.REQUEST_INTERVAL = 1.0             # APIレートリミット配慮のウェイト(秒)

        self.API_BASE = "https://api.inaturalist.org/v1/observations"

    def ensure_outdir(self):
        os.makedirs(self.OUT_DIR, exist_ok=True)

    def build_params(self, page: int):
        params = {
            "taxon_id": self.TAXON_ID,
            "quality_grade": self.QUALITY_GRADE,
            "photos": "true",
            "photo_license": ",".join(self.PHOTO_LICENSES),
            "page": page,
            "per_page": self.PER_PAGE,
            "order": "desc",
            "order_by": "created_at"
        }
        return params

    def get_original_url(self, photo: dict) -> str | None:
        """
        iNatのphotoオブジェクトからオリジナル画像URLを推定。
        一部は 'url' が square/medium/large を指すので original に置換。
        """
        # 場合によっては 'url' でなく 'original_url' が返ることもあるが、
        # 公式APIは通常 'url'。置換で original を狙う。
        url = photo.get("url")
        if not url:
            return None
        # 典型: .../photos/12345/square.jpg → .../photos/12345/original.jpg
        return url.replace("square.", "original.").replace("small.", "original.").replace("medium.", "original.").replace("large.", "original.")

    def download_file(self, url: str, dest_path: str, timeout: float = 30.0) -> bool:
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"[WARN] Download failed: {url} ({e})")
            return False

    def load_existing_meta(self):
        existing = set()
        if os.path.exists(self.META_CSV):
            with open(self.META_CSV, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # 一意キーとして photo_id を利用
                    if row.get("photo_id"):
                        existing.add(row["photo_id"])
        return existing

    def append_meta(self, rows):
        file_exists = os.path.exists(self.META_CSV)
        with open(self.META_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "photo_id",
                    "observation_id",
                    "photo_attribution",
                    "photo_license",
                    "observer_login",
                    "observation_url",
                    "photo_page_url",
                    "downloaded_path"
                ]
            )
            if not file_exists:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def main(self):
        self.ensure_outdir()
        seen_meta = self.load_existing_meta()

        page = 1
        pages_done = 0
        total_downloaded = 0

        while True:
            if self.MAX_PAGES is not None and pages_done >= self.MAX_PAGES:
                break

            params = self.build_params(page)
            query = f"{self.API_BASE}?{urlencode(params)}"
            print(f"[INFO] Fetching page {page} ... {query}")

            try:
                resp = requests.get(self.API_BASE, params=params, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                print(f"[ERROR] API request failed on page {page}: {e}")
                time.sleep(self.REQUEST_INTERVAL)
                continue

            data = resp.json()
            results = data.get("results", [])
            if not results:
                print("[INFO] No more results.")
                break

            rows_to_write = []
            for obs in results:
                obs_id = obs.get("id")
                observer_login = (obs.get("user") or {}).get("login")
                observation_url = f"https://www.inaturalist.org/observations/{obs_id}"

                photos = obs.get("photos") or []
                for p in photos:
                    photo_id = str(p.get("id"))
                    if not photo_id:
                        continue
                    if photo_id in seen_meta:
                        # 既にメタにある＝たぶんDL済み
                        continue

                    # 画像URL
                    url = self.get_original_url(p)
                    if not url:
                        continue

                    # 保存先
                    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
                    dest_path = os.path.join(self.OUT_DIR, f"photo_{photo_id}{ext}")

                    # 既存チェック（再実行時の再DL回避）
                    if os.path.exists(dest_path):
                        # メタだけ足りない場合があるので、rows_to_writeには入れる
                        pass
                    else:
                        ok = self.download_file(url, dest_path)
                        if not ok:
                            continue

                    # CC-BY の帰属情報確保（UI表示用の "attribution" が入っていることが多い）
                    attribution = p.get("attribution") or ""
                    license_code = p.get("license_code") or ""  # "cc-by" / "cc0" など
                    photo_page_url = f"https://www.inaturalist.org/photos/{photo_id}"

                    rows_to_write.append({
                        "photo_id": photo_id,
                        "observation_id": obs_id,
                        "photo_attribution": attribution,
                        "photo_license": license_code,
                        "observer_login": observer_login or "",
                        "observation_url": observation_url,
                        "photo_page_url": photo_page_url,
                        "downloaded_path": dest_path,
                    })
                    seen_meta.add(photo_id)
                    total_downloaded += 1

            if rows_to_write:
                self.append_meta(rows_to_write)
                print(f"[INFO] Saved metadata for {len(rows_to_write)} photos (total downloaded so far: {total_downloaded}).")
            else:
                print("[INFO] No new photos on this page.")

            pages_done += 1
            page += 1
            time.sleep(self.REQUEST_INTERVAL)

        print(f"[DONE] Download complete. Total photos processed: {len(seen_meta)}; newly downloaded: {total_downloaded}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('usage: python ./inaturalist_downloader.py (path_to_animal_list.tsv)')
        exit()
    path = sys.argv[1].strip()
    source = open(path, 'r', encoding='utf-8').readlines()
    title_line = source[0]
    data_lines = source[1:]
    for data in data_lines:
        name, scientific, url, taxon_id = [d.strip() for d in data.split('\t')]
        if taxon_id == '':
            print(f'{name}[{scientific}] taxon_id is empty...')
            exit()
    for data in data_lines:
        name, scientific, url, taxon_id = [d.strip() for d in data.split('\t')]
        inaturalist_downloader = INaturalistDownloader(taxon_id, f'images/{scientific}')
        inaturalist_downloader.main()
