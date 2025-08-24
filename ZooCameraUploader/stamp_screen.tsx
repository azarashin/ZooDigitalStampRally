import React, { useCallback, useState } from "react";
import {
  View,
  Text,
  ActivityIndicator,
  FlatList,
  SafeAreaView,
  Pressable, 
  Image, 
} from "react-native";
import { useFocusEffect } from "@react-navigation/native";
import { pickStampImage } from "./datetime_utility";
import { StampLog } from "./stamp_log";
import { loadStampList, addToday } from "./storage";
import { styles } from "./styles";
import { 
  STORAGE_KEY, 
  AUTO_STAMP_ON_VIEW, 
  ICON_ROOT, 
  RECENT_DAYS, 
  IMG_UNKNOWN} from "./constant_parameters";
import {STAMP_INFO_LIST, STAMP_COUNT} from "./variable_parameters";
import XTweetButton from "./x_tweet_button"

// --------- 画面：スタンプ閲覧 ---------
export function StampsScreen() {
  const [list, setList] = useState<StampLog[] | null>(null);

  useFocusEffect(
    useCallback(() => {
      let alive = true;
      (async () => {
        let arr = await loadStampList(STORAGE_KEY, STAMP_INFO_LIST);

        // 旧仕様の「表示のたび押す」を残したい場合はここで1件追加
        if (AUTO_STAMP_ON_VIEW) {
          const idx = arr.findIndex((s) => (s.acquiredDates ?? []).length === 0);
          if (idx !== -1) arr = await addToday(STORAGE_KEY, arr, idx);
        }

        if (alive) setList(arr);
      })();
      return () => {
        alive = false;
      };
    }, [])
  );

  if (!list) {
    return (
      <View style={styles.center}>
        <ActivityIndicator />
        <Text style={styles.gray}>読み込み中...</Text>
      </View>
    );
  }

  const stampedCount = list.filter((s) => s.acquiredDates.length > 0).length;

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <View style={{ padding: 16 }}>
        <Text style={styles.titleSm}>スタンプ {stampedCount}/{STAMP_COUNT}</Text>
        <Text style={styles.graySm}>
          ゲットして１週間以上経つと寝てしまうよ
        </Text>
        <XTweetButton
        />
      </View>

      <FlatList
        data={list.map((s, i) => ({ ...s, i }))}
        keyExtractor={(x) => String(x.i)}
        numColumns={3}
        contentContainerStyle={{ padding: 12 }}
        columnWrapperStyle={{ gap: 12 }}
        renderItem={({ item }) => (
          <StampCell
            id={item.id}
            name={item.name}
            dates={item.acquiredDates}
          />
        )}
      />
    </SafeAreaView>
  );
}

function StampCell({
  id,
  name,
  dates,
}: {
  id: string;
  name: string;
  dates: string[];
}) {
  const img = pickStampImage(ICON_ROOT, RECENT_DAYS, IMG_UNKNOWN, { name, id, acquiredDates: dates });
  const latest = dates.length ? dates[dates.length - 1] : null;

  return (
    <Pressable style={[styles.cell]}>
      <Image source={img} style={styles.cellImage} />
      <Text style={styles.cellName} numberOfLines={1}>
        {name}
      </Text>
      <Text style={styles.cellMeta}>
        {latest ? `${MakeShortTimestamp(latest)}` : "未入手"}
      </Text>
    </Pressable>
  );
}

const pad = (n: number) => String(n).padStart(2, "0");

// YYYY/MM/DD HH:mm
// 上記のフォーマットで日付を変換。
function MakeShortTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) throw new Error("Invalid date");
  return `${d.getUTCFullYear()}/${pad(d.getUTCMonth() + 1)}/${pad(d.getUTCDate())} ` +
         `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}
