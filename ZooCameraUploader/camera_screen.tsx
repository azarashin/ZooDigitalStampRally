
import React, { useEffect, useRef, useState } from "react";
import {
  View,
  TouchableOpacity,
  Text,
  Image,
  Alert,
  ActivityIndicator,
} from "react-native";
import * as MediaLibrary from "expo-media-library";
import * as ImageManipulator from "expo-image-manipulator";
import { CameraView, useCameraPermissions } from "expo-camera";
import {
  Gesture,
  GestureDetector,
  GestureHandlerRootView,
} from "react-native-gesture-handler";

import { AnalyzeResponse } from "./analyze_response";
import { StampLog } from "./stamp_log";
import { loadStampList, saveStampList, addToday, clamp } from "./storage";
import { styles } from "./styles";
import { 
  STORAGE_KEY, 
  API_ENDPOINT, 
} from "./constant_parameters";
import {STAMP_INFO_LIST, STAMP_COUNT} from "./variable_parameters";


// ----- 撮影画面（簡易プレビュー＋ボタン） -----
export function CameraScreen({ navigation }: any) {
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [mediaPermStatus, requestMediaPermission] =
    MediaLibrary.usePermissions();

  const cameraRef = useRef<CameraView | null>(null);

  const [capturedUri, setCapturedUri] = useState<string | null>(null);
  const [isBack, setIsBack] = useState<boolean>(true);
  const [flash, setFlash] = useState<"off" | "on" | "auto">("off");
  const [busy, setBusy] = useState<boolean>(false);

  const [zoom, setZoom] = useState(0); // 0..1
  const startZoomRef = useRef(0);
  const SENSITIVITY = 0.4;  // Pinch の感度

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
  const saveToLibrary = async (alert: boolean) => {
    if (!capturedUri) return;
    try {
      setBusy(true);
      await MediaLibrary.saveToLibraryAsync(capturedUri);
      if(alert)
      {
        Alert.alert("保存完了", "写真をライブラリに保存しました。");
      }
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
              Alert.alert(`${element.name}を見つけた！\n(自動保存)`);
            }
          });
          if(found == "")
          {
            // このメッセージが出たらデータの不整合を起こしている可能性あり。
            Alert.alert("[ERROR]動物を見つけられなかった…");
          }
          else
          {
            let list : StampLog[] = await loadStampList(STORAGE_KEY, STAMP_INFO_LIST);
            let index : number = -1; 
            for(let i: number = 0; i < list.length;i++)
            {
              if(list[i].id == found)
              {
                index = i; 
              }
            }
            if(index != -1)
            {
              list = await addToday(STORAGE_KEY, list, index); 
              saveToLibrary(false);
              navigation.navigate("Stamps");
            }
            else
            {
              Alert.alert("[ERROR]インデックスエラー");
            }
            await saveStampList(STORAGE_KEY, list);
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

  // ピンチでズーム
  const pinch = Gesture.Pinch()
    .onStart(() => {
      startZoomRef.current = zoom;
    })
    .onUpdate((e) => {
      // scale=1 を基準に相対的に拡大縮小
      const delta = Math.log(e.scale) * SENSITIVITY;
      const next = startZoomRef.current + delta;
      setZoom(clamp(next));
    });

  const resetCapturedUri = () => setCapturedUri(null);


  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <GestureDetector gesture={pinch}>
        <View style={styles.container}>
          {capturedUri ? (
            <View style={styles.previewWrap}>
              <Image source={{ uri: capturedUri }} style={styles.preview} />
              <View style={styles.row}>
                <TouchableOpacity
                  style={styles.btn}
                  onPress={() => saveToLibrary(true)}
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
                onPress={resetCapturedUri}
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
                zoom={zoom}
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
      </GestureDetector>
    </GestureHandlerRootView>
  );
}
