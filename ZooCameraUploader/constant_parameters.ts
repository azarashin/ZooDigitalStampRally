export const API_ENDPOINT = "http://192.168.0.38:3000/api/upload";
export const ICON_ROOT : string = "https://pit-creation.com/OpenDataHackathon2025/images/firefly" 
export const STORAGE_KEY = "stamps/v1";
export const BG_IMAGE = {
  // トップ画面の背景。
  uri: `${ICON_ROOT}/title/title.jpg`,
};

// 未入手のスタンプの態画像
export const IMG_UNKNOWN = { uri: `${ICON_ROOT}/unknown.png` };


// 画面表示時に自動で「今日付」を1件付与するか（以前の自動押印動作を残す場合は true）
export const AUTO_STAMP_ON_VIEW = false;

// 「7日以内」判定の閾値（日）。仕様変更しやすいよう定数化。
export const RECENT_DAYS = 7;
