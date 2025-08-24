import { StampLog } from "./stamp_log"


// --------- 日付ユーティリティ ---------
export function todayISO(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  const hour = String(d.getHours()).padStart(2, "0");
  const minute = String(d.getMinutes()).padStart(2, "0");
  const second = String(d.getSeconds()).padStart(2, "0");
  return `${y}-${m}-${day}T${hour}:${minute}:${second}Z`; // 端末TZのローカル日付ベース
}

export function daysDiff(afterISO: string, beforeISO: string): number {
  const before = new Date(beforeISO);
  const after = new Date(afterISO);
  const ms = after.getTime() - before.getTime();
  return ms / (24 * 60 * 60 * 1000);
}

// info → 状態画像（最近・ずっと前）を決定
export function pickStampImage(icon_root: string, recent_days : number, image_unknown : any, info: StampLog, refDateISO = todayISO()) {
  if (!info.acquiredDates?.length) return image_unknown;
  // 1つでも「refDateからRECENT_DAYS以内」があれば最近、なければずっと前
  const recent = info.acquiredDates.some((d) => {
    const diff = daysDiff(refDateISO, d);
    return diff <= recent_days;
  });
  return recent ? 
  {
    uri: `${icon_root}/animal_icon/${info.id}/active.jpg`
  } : 
  {
    uri: `${icon_root}/animal_icon/${info.id}/sleep.jpg`
  };
}
