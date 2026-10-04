import React from "react";

import {
  View,
  Image,
  StyleSheet
} from "react-native";


export default function ImagePreview({ uri }) {

  if (!uri) {
    return null;
  }

  return (
    <View style={styles.container}>

      <Image
        source={{ uri: uri }}
        style={styles.image}
        resizeMode="contain"
      />

    </View>
  );
}


const styles = StyleSheet.create({

  container: {
    width: "100%",
    height: 350,
    marginTop: 20,
    borderRadius: 16,
    overflow: "hidden",
    backgroundColor: "#F1F3F5"
  },

  image: {
    width: "100%",
    height: "100%"
  }

});
