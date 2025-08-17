import React, { useEffect, useRef, useState } from "react";
import {
  StyleSheet,
  View,
  TouchableOpacity,
  Text,
  Image,
  Alert,
  ActivityIndicator,
} from "react-native";
import { CameraView, CameraType, useCameraPermissions } from "expo-camera";
import * as MediaLibrary from "expo-media-library";
import * as ImageManipulator from "expo-image-manipulator";
import * as FileSystem from "expo-file-system";

const API_ENDPOINT = "http://192.168.0.37:3000/api/upload";

export default function App() {
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

      Alert.alert("送信完了", "サーバに画像を送信しました。");
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

const styles = StyleSheet.create({
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
  btnText: { color: "#fff", fontWeight: "700" },
  row: { flexDirection: "row" },
  center: { flex: 1, alignItems: "center", justifyContent: "center" },
});
