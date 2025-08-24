import React, { useCallback, useState, useEffect } from "react";
import { Alert, Button, View } from "react-native";
import * as WebBrowser from "expo-web-browser";
import * as Linking from "expo-linking";
import { loadStampList } from "./storage";
import { StampLog } from "./stamp_log";
import { useFocusEffect } from "@react-navigation/native";
import { daysDiff, todayISO } from "./datetime_utility";
import { 
  STORAGE_KEY, 
} from "./constant_parameters";
import {STAMP_INFO_LIST } from "./variable_parameters";

import { PLACE, PLACE_URL, PLACE_X_ACCOUNT } from "./variable_parameters";

type OpenTweetOptions = {
  text?: string;         // 本文
  url?: string;          // 一緒に添付するURL
  hashtags?: string[];   // ハッシュタグ。["Unity", "Expo"] のように
  via?: string;          // "your_account"（任意）
  related?: string[];    // 関連アカウント（任意）
};

const buildIntentUrl = (opts: OpenTweetOptions) => {
  const params = new URLSearchParams();

  if (opts.text) params.set("text", opts.text);
  if (opts.url) params.set("url", opts.url);
  if (opts.hashtags?.length) params.set("hashtags", opts.hashtags.join(","));
  if (opts.via) params.set("via", opts.via);
  if (opts.related?.length) params.set("related", opts.related.join(","));

  // x.com と twitter.com はどちらも有効。まず x.com を試して、だめなら twitter.com を使う。
  const primary = `https://x.com/intent/tweet?${params.toString()}`;
  const fallback = `https://twitter.com/intent/tweet?${params.toString()}`;
  return { primary, fallback };
};

async function openXComposer(opts: OpenTweetOptions) {
  try {
    const { primary, fallback } = buildIntentUrl(opts);

    // Universal Linkとしてまず直接開く（アプリがあればアプリに遷移することが多い）
    const canOpen = await Linking.canOpenURL(primary);
    if (canOpen) {
      await Linking.openURL(primary);
      return;
    }

    // ブラウザで開く（アプリがなくてもOK）
    const result = await WebBrowser.openBrowserAsync(primary);
    if (result.type === "cancel") {
      // 稀に x.com がブロックされているケースに対するフォールバック
      await WebBrowser.openBrowserAsync(fallback);
    }
  } catch (e) {
    console.error(e);
    Alert.alert("エラー", "Xの投稿画面を開けませんでした。ネットワーク状態をご確認ください。");
  }
}

function isToday(today: string, log: StampLog): boolean {
  if (log.acquiredDates.length === 0) 
  {
    return false;
  }
  for (let i = 0; i < log.acquiredDates.length; i++) {
    const length = daysDiff(today, log.acquiredDates[i]);
    if(length <= 1.0)
    {
      return true; 
    }
  }
  return false; 
}

export default function XTweetButton() {

  const onPress = useCallback(async () => {
    let text = "";
    const today = todayISO();
    let stampedCount = 0;
    let list = await loadStampList(STORAGE_KEY, STAMP_INFO_LIST);
    if(list != null)
    {
      stampedCount = list.filter(s => isToday(today, s)).length;
    }
    if(stampedCount === 0)
    {
      text = `今、${PLACE}にいます。\nこれからスタンプを集めます！`
    }
    else
    {
      text = `${PLACE}で${stampedCount}個のスタンプを獲得しました！`
    }
    openXComposer({
      text: text, // 本文
      url: PLACE_URL,              // 一緒に添付するURL
      hashtags: [ PLACE ],    // ハッシュタグ(任意)
      via: PLACE_X_ACCOUNT,                  // "your_account"（任意）
      related: [ PLACE_X_ACCOUNT ],     // 関連アカウント（任意）
    });
  }, []);
  return (
    <View style={{ padding: 16 }}>
      <Button title="Xでポストする" onPress={onPress} />
    </View>
  );
}