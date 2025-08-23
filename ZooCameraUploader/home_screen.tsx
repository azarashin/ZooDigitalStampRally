import { styles } from "./styles";
import {
  View,
  Text,
  ImageBackground,
  Pressable,
  SafeAreaView,
} from "react-native";
import { deleteLog } from "./storage";
import {BG_IMAGE, STORAGE_KEY} from "./constant_parameters";
import {STAMP_INFO_LIST} from "./variable_parameters";

// ----- トップ画面 -----
export function HomeScreen({ navigation }: any) {
  return (
    <ImageBackground source={BG_IMAGE} style={styles.bg} resizeMode="cover">
      <SafeAreaView style={styles.overlay}>
        
        {/*
        <Text style={styles.title}>デジタルスタンプラリー</Text>
        */}
        <View style={styles.homeButtons}>
          <Pressable
            style={styles.primaryBtn}
            onPress={() => navigation.navigate("Camera")}
          >
            <Text style={styles.btnText}>撮影</Text>
          </Pressable>
          <Pressable
            style={[styles.primaryBtn, styles.secondaryBtn, styles.mt12]}
            onPress={() => navigation.navigate("Stamps")}
          >
            <Text style={styles.btnText}>スタンプ閲覧</Text>
          </Pressable>
          <Pressable
            style={[styles.primaryBtn, styles.secondaryBtn, styles.mt12]}
            onPress={async () => await deleteLog(STORAGE_KEY, STAMP_INFO_LIST)}
          >
            <Text style={styles.btnText}>【開発用】ログを削除する</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    </ImageBackground>
  );
}
