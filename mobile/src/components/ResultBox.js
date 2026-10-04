import React from "react";

import {
  View,
  Text,
  TextInput,
  StyleSheet
} from "react-native";


export default function ResultBox({
  value,
  onChangeText
}) {

  return (
    <View style={styles.container}>

      <Text style={styles.title}>
        Extracted Answer
      </Text>

      <TextInput
        value={value}
        onChangeText={onChangeText}
        multiline={true}
        textAlignVertical="top"
        placeholder="The handwriting extracted from the examination page will appear here..."
        style={styles.input}
      />

    </View>
  );
}


const styles = StyleSheet.create({

  container: {
    marginTop: 25
  },

  title: {
    fontSize: 20,
    fontWeight: "700",
    marginBottom: 10
  },

  input: {
    minHeight: 280,
    borderWidth: 1,
    borderColor: "#D0D0D0",
    borderRadius: 14,
    padding: 15,
    fontSize: 16,
    backgroundColor: "#FFFFFF"
  }

});
