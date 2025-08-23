import { pickStampImage, todayISO } from "./datetime_utility";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { StampLog } from "./stamp_log";
import { StampInfo } from "./stamp_info";
import { Alert } from "react-native";

// --------- ストレージ ---------
export async function loadStampList(storage_key: string, stamp_info_list : StampInfo[]): Promise<StampLog[]> {
  const raw = await AsyncStorage.getItem(storage_key);
  if (raw) {
    try {
      const parsed = JSON.parse(raw) as StampLog[];
      return normalizeList(parsed, stamp_info_list);
    } catch {
      // 壊れていたら初期化
    }
  }
  // 初期化（未入手）
  const init: StampLog[] = Array.from({ length: stamp_info_list.length }, (_, i) => ({
    id: stamp_info_list[i].id,
    name: stamp_info_list[i].name,
    acquiredDates: [],
  }));
  await AsyncStorage.setItem(storage_key, JSON.stringify(init));
  return init;
}

function normalizeList(list: StampLog[], stamp_info_list : StampInfo[]): StampLog[] {
  // STAMP_COUNT を変更した場合に合わせる
  const base = Array.from({ length: stamp_info_list.length }, (_, i) => ({
    id: stamp_info_list[i].id,
    name: stamp_info_list[i].name,
    acquiredDates: [] as string[],
  }));
  for (let i = 0; i < Math.min(list.length, base.length); i++) {
    base[i] = {
      id: list[i]?.id ?? base[i].id,
      name: list[i]?.name ?? base[i].name,
      acquiredDates: Array.isArray(list[i]?.acquiredDates) ? list[i].acquiredDates : [],
    };
  }
  return base;
}

export async function deleteLog(storage_key: string, stamp_info_list : StampInfo[]) {
  const init: StampLog[] = Array.from({ length: stamp_info_list.length }, (_, i) => ({
    id: stamp_info_list[i].id,
    name: stamp_info_list[i].name,
    acquiredDates: [],
  }));
  await AsyncStorage.setItem(storage_key, JSON.stringify(init));
  Alert.alert("記録を削除しました");
}

export async function saveStampList(storage_key: string, list: StampLog[]) {
  await AsyncStorage.setItem(storage_key, JSON.stringify(list));
}

// 任意のインデックスに「今日」を追加
export async function addToday(storage_key: string, list: StampLog[], index: number): Promise<StampLog[]> {
  const d = todayISO();
  const next = [...list];
  const arr = new Set(next[index].acquiredDates ?? []);
  arr.add(d);
  next[index] = { ...next[index], acquiredDates: Array.from(arr).sort() };
  await saveStampList(storage_key, next);
  return next;
}

export const clamp = (v: number, min = 0, max = 1) => Math.min(max, Math.max(min, v));
