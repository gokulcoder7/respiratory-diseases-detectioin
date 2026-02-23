import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# 1. Automatic Path & Custom Layer Setup
# This forces Python to look in the same folder as this script file
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "gru_model.h5")

# We must keep this class to avoid the 'time_major' ValueError from your original code
class FixedGRU(tf.keras.layers.GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# 2. Load Model with the Fix
try:
    gru_model = tf.keras.models.load_model(
        model_path,
        custom_objects={'GRU': FixedGRU}
    )
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop() # Stop the app if model fails

classes = ["COPD" ,"Bronchiolitis ", "Pneumoina", "URTI", "Healthy"]

# 3. Core Prediction Logic
def gru_diagnosis_prediction(test_audio):
    data_x, sampling_rate = librosa.load(test_audio)
    # Original processing: stretch and MFCC
    # data_x = librosa.effects.time_stretch(data_x, rate=1.2)
    features = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=52).T, axis=0)
    features = features.reshape(1, 52)
    
    test_pred = gru_model.predict(np.expand_dims(features, axis=1))
    classpreds = classes[np.argmax(test_pred[0], axis=1)[0]]
    confidence = test_pred.T[test_pred[0].mean(axis=0).argmax()].mean()
    return classpreds, confidence

# 4. Streamlit Interface
st.title("RESPIRATORY DISEASE DETECTION")
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav","audio"])

if uploaded_file is not None:
    if st.button("Generate Prediction"):
        prediction_type, accuracy = gru_diagnosis_prediction(uploaded_file)
        st.success(f"Diagnosis: {prediction_type}")
        st.info(f"Confidence: {accuracy:.4f}")