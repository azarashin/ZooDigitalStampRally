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

// ===================== 設定 =====================
const STAMP_COUNT = 12; // ← スタンプ数（N）
const STORAGE_KEY = "stamps/v1";
const BG_IMAGE = {
  // ← トップ画面の背景。任意のURLに変えてOK（ローカルなら require("./assets/bg.jpg")）
  uri: "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=1600",
};
// ===============================================

type RootStackParamList = {
  Home: undefined;
  Camera: undefined;
  Stamps: undefined;
};

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

      Alert.alert(d.predict?.best_label + "\n" + d.predict?.best_confidence);
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
              <Text style={styles.btnText}>API送信</Text>
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


// ----- スタンプ閲覧画面 -----
function StampsScreen() {
  const [stamps, setStamps] = useState<boolean[] | null>(null);

  // 画面表示のたびに“次の空き”にスタンプを押して保存
  useFocusEffect(
    useCallback(() => {
      let alive = true;
      (async () => {
        const arr = await loadStamps();
        const idx = arr.findIndex((v) => !v);
        if (idx !== -1) {
          arr[idx] = true;
          await saveStamps(arr);
        }
        if (alive) setStamps(arr);
      })();
      return () => {
        alive = false;
      };
    }, [])
  );

  if (!stamps) {
    return (
      <View style={styles.center}>
        <ActivityIndicator />
        <Text style={styles.gray}>読み込み中...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <View style={{ padding: 16 }}>
        <Text style={styles.titleSm}>
          スタンプ {stamps.filter(Boolean).length}/{STAMP_COUNT}
        </Text>
      </View>
      <FlatList
        data={stamps.map((v, i) => ({ i, stamped: v }))}
        keyExtractor={(x) => String(x.i)}
        numColumns={4}
        contentContainerStyle={{ padding: 12 }}
        columnWrapperStyle={{ gap: 12 }}
        renderItem={({ item }) => <StampCell index={item.i} stamped={item.stamped} />}
      />
    </SafeAreaView>
  );
}

function StampCell({ index, stamped }: { index: number; stamped: boolean }) {
  return (
    <View style={[styles.cell, stamped ? styles.cellOn : styles.cellOff]}>
      <Text style={[styles.cellNum, stamped && styles.cellNumOn]}>
        {index + 1}
      </Text>
      <Text style={styles.cellIcon}>{stamped ? "✅" : "☆"}</Text>
    </View>
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
  bottomBar: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    padding: 12,
    backgroundColor: "rgba(0,0,0,0.4)",
  },
  cell: {
    flex: 1,
    aspectRatio: 1,
    borderRadius: 16,
    borderWidth: 2,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 12,
  },
  cellOff: { borderColor: "#bbb", backgroundColor: "#fafafa" },
  cellOn: { borderColor: "#2ecc71", backgroundColor: "#eafff1" },
  cellNum: { fontSize: 16, color: "#555", fontWeight: "700" },
  cellNumOn: { color: "#2e7d32" },
  cellIcon: { fontSize: 28, marginTop: 6 },
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
