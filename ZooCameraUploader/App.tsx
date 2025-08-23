import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  StyleSheet,
  View,
  TouchableOpacity,
  Text,
  Image,
  Alert,
  ActivityIndicator,
  FlatList,
  ImageBackground,
  Pressable,
  SafeAreaView,
} from "react-native";
import { NavigationContainer, useFocusEffect } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as MediaLibrary from "expo-media-library";
import * as ImageManipulator from "expo-image-manipulator";
import { CameraView, useCameraPermissions } from "expo-camera";

const API_ENDPOINT = "http://192.168.0.37:3000/api/upload";

type AnalyzeResponse = {
  ok: boolean;
  file?: {
    size: number;
    mimetype: string;
  };
predict?: {
    best_label: string;
    best_confidence: number;
    topk?: [{
       label: string; 
       confidence: number;
      }];
  };
  meta?: {
    userId?: string;
  };
  bytes?: number;
};

type StampInfo = {
  id: string;
  name: string;

};

const ICON_ROOT : string = "https://pit-creation.com/OpenDataHackathon2025/images/firefly" 

// ===================== 設定 =====================
const STAMP_INFO_LIST : StampInfo[] = [
  {id: "Acinonyx jubatus", name: "チーター"}, 
  {id: "Anser cygnoides", name: "サカツラガン"}, 
  {id: "Bubalus bubalis", name: "スイギュウ"}, 
  {id: "Capricornis swinhoei", name: "タイワンカモシカ"}, 
  {id: "Elephas maximus", name: "アジアゾウ"}, 
  {id: "Giraffa giraffa giraffa", name: "ミナミキリン"}, 
  {id: "Hylobates lar", name: "シロテテナガザル"}, 
  {id: "Leptailurus serval", name: "サーバル"}, 
  {id: "Loxodonta africana", name: "アフリカゾウ"}, 
  {id: "Macaca fuscata", name: "ニホンザル"}, 
  {id: "Oryx leucoryx", name: "アラビアオリックス"}, 
  {id: "Osphranter rufus", name: "アカカンガルー"}, 
  {id: "Pan troglodytes", name: "チンパンジー"}, 
  {id: "Phascolarctos cinereus", name: "コアラ"}, 
  {id: "Pongo pygmaeus", name: "ボルネオオランウータン"}, 
  {id: "Rangifer tarandus", name: "トナカイ"}, 
  {id: "Rhinoceros unicornis", name: "インドサイ"}, 

];
const STAMP_COUNT = STAMP_INFO_LIST.length; // ← スタンプ数（N）
const STORAGE_KEY = "stamps/v1";
const BG_IMAGE = {
  // ← トップ画面の背景。任意のURLに変えてOK（ローカルなら require("./assets/bg.jpg")）
  uri: "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=1600",
};
// 状態画像（A=未入手 / B=7日以内 / C=7日より前）
// ※ 実運用は下記を require("./assets/a.png") などローカルへ差し替えてください
const IMG_UNKNOWN = { uri: `${ICON_ROOT}/unknown.png` };


// 画面表示時に自動で「今日付」を1件付与するか（以前の自動押印動作を残す場合は true）
const AUTO_STAMP_ON_VIEW = false;

// 「7日以内」判定の閾値（日）。仕様変更しやすいよう定数化。
const RECENT_DAYS = 7;
// ===============================================

type RootStackParamList = {
  Home: undefined;
  Camera: undefined;
  Stamps: undefined;
};

type StampLog = {
  id: string; 
  name: string; 
  acquiredDates: string[];
}; // ISO日付(YYYY-MM-DD)配列

const Stack = createNativeStackNavigator<RootStackParamList>();

// ----- ストレージヘルパ -----
async function loadStamps(): Promise<boolean[]> {
  const raw = await AsyncStorage.getItem(STORAGE_KEY);
  let arr: boolean[] =
    raw ? (JSON.parse(raw) as boolean[]) : Array(STAMP_COUNT).fill(false);
  // 長さの補正（設定を後から変えた場合に備える）
  if (arr.length !== STAMP_COUNT) {
    const next = Array(STAMP_COUNT).fill(false);
    for (let i = 0; i < Math.min(arr.length, STAMP_COUNT); i++) next[i] = arr[i];
    arr = next;
  }
  return arr;
}
async function saveStamps(arr: boolean[]) {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(arr));
}

// --------- 日付ユーティリティ ---------
function todayISO(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`; // 端末TZのローカル日付ベース
}

function daysDiff(aISO: string, bISO: string): number {
  const a = new Date(aISO + "T00:00:00");
  const b = new Date(bISO + "T00:00:00");
  const ms = b.getTime() - a.getTime();
  return Math.floor(ms / (24 * 60 * 60 * 1000));
}

// info → 状態画像（A/B/C）を決定
function pickStampImage(info: StampLog, refDateISO = todayISO()) {
  if (!info.acquiredDates?.length) return IMG_UNKNOWN;
  // 1つでも「refDateからRECENT_DAYS以内」があればB、なければC
  const recent = info.acquiredDates.some((d) => {
    const diff = Math.abs(daysDiff(d, refDateISO));
    return diff <= RECENT_DAYS;
  });
  return recent ? 
  {
    uri: `${ICON_ROOT}/animal_icon/${info.id}/active.jpg`
  } : 
  {
    uri: `${ICON_ROOT}/animal_icon/${info.id}/sleep.jpg`
  };
}

// --------- ストレージ ---------
async function loadStampList(): Promise<StampLog[]> {
  const raw = await AsyncStorage.getItem(STORAGE_KEY);
  if (raw) {
    try {
      const parsed = JSON.parse(raw) as StampLog[];
      return normalizeList(parsed);
    } catch {
      // 壊れていたら初期化
    }
  }
  // 初期化（未入手）
  const init: StampLog[] = Array.from({ length: STAMP_COUNT }, (_, i) => ({
    id: STAMP_INFO_LIST[i].id,
    name: STAMP_INFO_LIST[i].name,
    acquiredDates: [],
  }));
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(init));
  return init;
}

function normalizeList(list: StampLog[]): StampLog[] {
  // STAMP_COUNT を変更した場合に合わせる
  const base = Array.from({ length: STAMP_COUNT }, (_, i) => ({
    id: STAMP_INFO_LIST[i].id,
    name: STAMP_INFO_LIST[i].name,
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

async function saveStampList(list: StampLog[]) {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

// 任意のインデックスに「今日」を追加
async function addToday(list: StampLog[], index: number): Promise<StampLog[]> {
  const d = todayISO();
  const next = [...list];
  const arr = new Set(next[index].acquiredDates ?? []);
  arr.add(d);
  next[index] = { ...next[index], acquiredDates: Array.from(arr).sort() };
  await saveStampList(next);
  return next;
}

// ----- トップ画面 -----
function HomeScreen({ navigation }: any) {
  return (
    <ImageBackground source={BG_IMAGE} style={styles.bg} resizeMode="cover">
      <SafeAreaView style={styles.overlay}>
        <Text style={styles.title}>デジタルスタンプラリー</Text>
        <View style={{ height: 24 }} />
        <Pressable
          style={styles.primaryBtn}
          onPress={() => navigation.navigate("Camera")}
        >
          <Text style={styles.btnText}>撮影</Text>
        </Pressable>
        <Pressable
          style={[styles.primaryBtn, styles.secondaryBtn]}
          onPress={() => navigation.navigate("Stamps")}
        >
          <Text style={styles.btnText}>スタンプ閲覧</Text>
        </Pressable>
      </SafeAreaView>
    </ImageBackground>
  );
}

// ----- 撮影画面（簡易プレビュー＋ボタン） -----
function CameraScreen({ navigation }: any) {
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [mediaPermStatus, requestMediaPermission] =
    MediaLibrary.usePermissions();

  const cameraRef = useRef<CameraView | null>(null);

  const [capturedUri, setCapturedUri] = useState<string | null>(null);
  const [isBack, setIsBack] = useState<boolean>(true);
  const [flash, setFlash] = useState<"off" | "on" | "auto">("off");
  const [busy, setBusy] = useState<boolean>(false);

  useEffect(() => {
    (async () => {
      if (!cameraPermission?.granted) await requestCameraPermission();
      if (!mediaPermStatus?.granted) await requestMediaPermission();
    })();
  }, []);

  if (!cameraPermission || !mediaPermStatus) {
    return (
      <View style={styles.center}>
        <ActivityIndicator />
        <Text style={{ marginTop: 8 }}>権限確認中...</Text>
      </View>
    );
  }

  if (!cameraPermission.granted) {
    return (
      <View style={styles.center}>
        <Text>カメラ権限が必要です。</Text>
        <TouchableOpacity style={styles.btn} onPress={requestCameraPermission}>
          <Text style={styles.btnText}>権限を許可</Text>
        </TouchableOpacity>
      </View>
    );
  }

  /** 撮影 */
  const takePhoto = async () => {
    if (!cameraRef.current) return;
    try {
      setBusy(true);
      const photo = await cameraRef.current.takePictureAsync({
        quality: 1,
        skipProcessing: false,
      });
      setCapturedUri(photo.uri);
    } catch (e: any) {
      Alert.alert("撮影エラー", e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  /** 端末のフォトライブラリに保存 */
  const saveToLibrary = async () => {
    if (!capturedUri) return;
    try {
      setBusy(true);
      await MediaLibrary.saveToLibraryAsync(capturedUri);
      Alert.alert("保存完了", "写真をライブラリに保存しました。");
    } catch (e: any) {
      Alert.alert("保存エラー", e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  /** アップロード用にリサイズ・圧縮 */
  const prepareForUpload = async (uri: string): Promise<string> => {
    const manip = await ImageManipulator.manipulateAsync(
      uri,
      [{ resize: { width: 1280 } }],
      { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
    );
    return manip.uri;
  };

  /** Web API へ送信 */
  const uploadToApi = async () => {
    if (!capturedUri) return;
    try {
      setBusy(true);
      const resizedUri = await prepareForUpload(capturedUri);

      const name = "photo.jpg";
      const type = "image/jpeg";

      const form = new FormData();
      form.append("file", {
        uri: resizedUri,
        name,
        type,
      } as any); // RN独自型なので as any
      form.append("userId", "12345");

      const endpoint = API_ENDPOINT;

      const res = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "multipart/form-data",
        },
        body: form,
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      // JSON 取得
      const data = (await res.json()) as unknown;

      // 型アサーション（軽量）：サーバの契約が決まっている前提
      const d = data as AnalyzeResponse;

      if(d.predict)
      {
        if(d.predict.best_label == "-1")
        {
          Alert.alert("動物を見つけられなかった…");
        }
        else
        {
          let found: string = "";
          STAMP_INFO_LIST.forEach(element => {
            if(d.predict && element.id == d.predict.best_label)
            {
              found = element.id; 
              
              Alert.alert(`${element.name}を見つけた！`);
            }
          });
          if(found == "")
          {
            // このメッセージが出たらデータの不整合を起こしている可能性あり。
            Alert.alert("[ERROR]動物を見つけられなかった…");
          }
          else
          {
            let list : StampLog[] = await loadStampList();
            let index : number = -1; 
            for(let i: number = 0; i < list.length;i++)
            {
              if(list[i].id == found)
              {
                index = i; 
              }
            }
            list = await addToday(list, index); 
            await saveStampList(list);
          }
        }
      }
      else
      {
        // 応答にpredict が含まれていない。サーバ側での応答内容が壊れている。
        Alert.alert("[ERROR]");
      }

      //Alert.alert(d.predict?.best_label + "\n" + d.predict?.best_confidence);
    } catch (e: any) {
      Alert.alert("送信エラー", e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  const reset = () => setCapturedUri(null);

  return (
    <View style={styles.container}>
      {capturedUri ? (
        <View style={styles.previewWrap}>
          <Image source={{ uri: capturedUri }} style={styles.preview} />
          <View style={styles.row}>
            <TouchableOpacity
              style={styles.btn}
              onPress={saveToLibrary}
              disabled={busy}
            >
              <Text style={styles.btnText}>ライブラリ保存</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.btn}
              onPress={uploadToApi}
              disabled={busy}
            >
              <Text style={styles.btnText}>スタンプ入手</Text>
            </TouchableOpacity>
          </View>
          <TouchableOpacity
            style={[styles.btn, styles.secondary]}
            onPress={reset}
            disabled={busy}
          >
            <Text style={styles.btnText}>撮り直す</Text>
          </TouchableOpacity>
          {busy && <ActivityIndicator style={{ marginTop: 12 }} />}
        </View>
      ) : (
        <>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={isBack ? "back" : "front"}
            flash={flash}
          />
          <View style={styles.controls}>
            <TouchableOpacity
              style={[styles.smallBtn, styles.secondary]}
              onPress={() => setIsBack((v) => !v)}
              disabled={busy}
            >
              <Text style={styles.smallBtnText}>
                {isBack ? "前面" : "背面"}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.shutter}
              onPress={takePhoto}
              disabled={busy}
            />
            <TouchableOpacity
              style={[styles.smallBtn, styles.secondary]}
              onPress={() =>
                setFlash((f) =>
                  f === "off" ? "on" : f === "on" ? "auto" : "off"
                )
              }
              disabled={busy}
            >
              <Text style={styles.smallBtnText}>Flash: {flash}</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
    </View>
  );
}


// --------- 画面：スタンプ閲覧 ---------
function StampsScreen() {
  const [list, setList] = useState<StampLog[] | null>(null);

  useFocusEffect(
    useCallback(() => {
      let alive = true;
      (async () => {
        let arr = await loadStampList();

        // 旧仕様の「表示のたび押す」を残したい場合はここで1件追加
        if (AUTO_STAMP_ON_VIEW) {
          const idx = arr.findIndex((s) => (s.acquiredDates ?? []).length === 0);
          if (idx !== -1) arr = await addToday(arr, idx);
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
          ルール：未入手=A, 直近{RECENT_DAYS}日以内=B, それより前=C
        </Text>
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
            onAddToday={async () => {
              const next = await addToday(list, item.i);
              setList(next);
              Alert.alert("スタンプ更新", `${item.name} に本日の日付を追加しました。`);
            }}
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
  onAddToday,
}: {
  id: string;
  name: string;
  dates: string[];
  onAddToday: () => void;
}) {
  const img = pickStampImage({ name, id, acquiredDates: dates });
  const latest = dates.length ? dates[dates.length - 1] : null;

  return (
    <Pressable style={[styles.cell]} onLongPress={onAddToday}>
      <Image source={img} style={styles.cellImage} />
      <Text style={styles.cellName} numberOfLines={1}>
        {name}
      </Text>
      <Text style={styles.cellMeta}>
        {latest ? `最終入手: ${latest}` : "未入手"}
      </Text>
      <Text style={styles.cellHint}>（長押しで今日を追加）</Text>
    </Pressable>
  );
}
// ----- ルート -----
export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ title: "トップ" }}
        />
        <Stack.Screen
          name="Camera"
          component={CameraScreen}
          options={{ title: "撮影" }}
        />
        <Stack.Screen
          name="Stamps"
          component={StampsScreen}
          options={{ title: "スタンプ閲覧" }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

// ----- スタイル -----
const styles = StyleSheet.create({
  bg: { flex: 1 },
  overlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.35)",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  title: {
    color: "#fff",
    fontSize: 28,
    fontWeight: "800",
    textShadowColor: "rgba(0,0,0,0.4)",
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  titleSm: { fontSize: 18, fontWeight: "700" },
  primaryBtn: {
    backgroundColor: "#007aff",
    paddingVertical: 14,
    paddingHorizontal: 18,
    borderRadius: 14,
    minWidth: 220,
    alignItems: "center",
  },
  secondaryBtn: { backgroundColor: "#444" },
  btnText: { color: "#fff", fontWeight: "700" },
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24 },
  gray: { color: "#666", marginTop: 8 },
  graySm: { color: "#666" },
  cell: {
    flex: 1,
    aspectRatio: 1,
    borderRadius: 16,
    borderWidth: 2,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 12,
  },
  cellImage: { width: "70%", height: "55%", resizeMode: "contain" },
  cellName: { marginTop: 6, fontWeight: "700" },
  cellMeta: { fontSize: 12, color: "#666" },
  cellHint: { fontSize: 10, color: "#aaa", marginTop: 2 },
  container: { flex: 1, backgroundColor: "#000" },
  camera: { flex: 1 },
  controls: {
    position: "absolute",
    bottom: 28,
    width: "100%",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-around",
  },
  shutter: {
    width: 78,
    height: 78,
    borderRadius: 39,
    backgroundColor: "#fff",
    borderWidth: 4,
    borderColor: "#ddd",
  },
  smallBtn: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 12,
    backgroundColor: "#222",
  },
  smallBtnText: { color: "#fff", fontWeight: "600" },
  previewWrap: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#111",
  },
  preview: { width: "100%", height: "75%", resizeMode: "contain" },
  btn: {
    backgroundColor: "#007aff",
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 14,
    margin: 8,
  },
  secondary: { backgroundColor: "#444" },
  row: { flexDirection: "row" },
});
