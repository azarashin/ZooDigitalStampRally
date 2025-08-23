import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { RootStackParamList } from "./root_stack_param_list";

import { HomeScreen } from "./home_screen";
import { CameraScreen } from "./camera_screen";
import { StampsScreen } from "./stamp_screen";

const Stack = createNativeStackNavigator<RootStackParamList>();

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
