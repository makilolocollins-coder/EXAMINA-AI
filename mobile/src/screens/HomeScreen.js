import React, { useState } from "react";

import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  ScrollView
} from "react-native";

import * as ImagePicker from "expo-image-picker";

import ImagePreview from "../components/ImagePreview";
import ResultBox from "../components/ResultBox";

import { performOCR } from "../services/ocrService";


export default function HomeScreen() {

  const [imageUri, setImageUri] = useState(null);

  const [ocrText, setOcrText] = useState("");

  const [loading, setLoading] = useState(false);


  // ========================================================
  // SELECT IMAGE
  // ========================================================

  async function pickImage() {

    const permission =
      await ImagePicker.requestMediaLibraryPermissionsAsync();


    if (!permission.granted) {

      Alert.alert(
        "Permission required",
        "Examina AI needs access to your photos."
      );

      return;
    }


    const result =
      await ImagePicker.launchImageLibraryAsync({

        mediaTypes: ["images"],

        allowsEditing: false,

        quality: 1

      });


    if (!result.canceled) {

      setImageUri(
        result.assets[0].uri
      );

      setOcrText("");
    }
  }


  // ========================================================
  // TAKE PHOTO
  // ========================================================

  async function takePhoto() {

    const permission =
      await ImagePicker.requestCameraPermissionsAsync();


    if (!permission.granted) {

      Alert.alert(
        "Permission required",
        "Examina AI needs access to your camera."
      );

      return;
    }


    const result =
      await ImagePicker.launchCameraAsync({

        allowsEditing: false,

        quality: 1

      });


    if (!result.canceled) {

      setImageUri(
        result.assets[0].uri
      );

      setOcrText("");
    }
  }


  // ========================================================
  // OCR
  // ========================================================

  async function readHandwriting() {

    if (!imageUri) {

      Alert.alert(
        "No examination page",
        "Please upload or photograph an examination page first."
      );

      return;
    }


    try {

      setLoading(true);

      setOcrText("");


      const result =
        await performOCR(imageUri);


      if (result.text) {

        setOcrText(result.text);

      } else {

        setOcrText(
          "No handwriting was detected."
        );
      }

    }

    catch (error) {

      Alert.alert(
        "OCR Error",
        error.message
      );

    }

    finally {

      setLoading(false);
    }
  }


  // ========================================================
  // UI
  // ========================================================

  return (

    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
    >

      <Text style={styles.title}>
        Examina AI
      </Text>

      <Text style={styles.subtitle}>
        Intelligent Examination Assistant
      </Text>


      <View style={styles.buttonRow}>

        <TouchableOpacity
          style={styles.cameraButton}
          onPress={takePhoto}
        >

          <Text style={styles.whiteButtonText}>
            📷 Take Photo
          </Text>

        </TouchableOpacity>


        <TouchableOpacity
          style={styles.uploadButton}
          onPress={pickImage}
        >

          <Text style={styles.darkButtonText}>
            📁 Upload
          </Text>

        </TouchableOpacity>

      </View>


      <ImagePreview
        uri={imageUri}
      />


      {imageUri && (

        <TouchableOpacity
          style={styles.readButton}
          onPress={readHandwriting}
          disabled={loading}
        >

          {loading ? (

            <View style={styles.loadingContainer}>

              <ActivityIndicator
                color="#FFFFFF"
              />

              <Text style={styles.whiteButtonText}>
                Reading handwriting...
              </Text>

            </View>

          ) : (

            <Text style={styles.whiteButtonText}>
              🔍 Read Handwriting
            </Text>

          )}

        </TouchableOpacity>

      )}


      {ocrText !== "" && (

        <ResultBox
          value={ocrText}
          onChangeText={setOcrText}
        />

      )}

    </ScrollView>
  );
}


// ==========================================================
// STYLES
// ==========================================================

const styles = StyleSheet.create({

  screen: {
    flex: 1,
    backgroundColor: "#FFFFFF"
  },

  content: {
    padding: 20,
    paddingTop: 60,
    paddingBottom: 60
  },

  title: {
    fontSize: 34,
    fontWeight: "800",
    color: "#111111"
  },

  subtitle: {
    fontSize: 16,
    color: "#666666",
    marginTop: 6
  },

  buttonRow: {
    flexDirection: "row",
    marginTop: 30,
    gap: 10
  },

  cameraButton: {
    flex: 1,
    padding: 17,
    borderRadius: 14,
    backgroundColor: "#111111",
    alignItems: "center"
  },

  uploadButton: {
    flex: 1,
    padding: 17,
    borderRadius: 14,
    backgroundColor: "#EEEEEE",
    alignItems: "center"
  },

  readButton: {
    marginTop: 20,
    padding: 18,
    borderRadius: 14,
    backgroundColor: "#1769FF",
    alignItems: "center"
  },

  loadingContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10
  },

  whiteButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "700"
  },

  darkButtonText: {
    color: "#111111",
    fontSize: 16,
    fontWeight: "700"
  }

});
