import streamlit as st
import pandas as pd
import numpy as np
import pyedflib
import neurokit2 as nk
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import mne
edf_data = mne.io.read_raw_edf("your_file.edf", preload=True)


# 📌 Load the trained model, scaler, and selected features
model = joblib.load("ecg_model_reduced.pkl")
scaler = joblib.load("scaler_reduced.pkl")
selected_features = joblib.load("selected_features.pkl")

# 📌 Streamlit App Title
st.title("🔬 ECG Prediction with R-Peak Detection & Severity Level")

# 📌 Function to Extract HRV Features & Detect R-Peaks
def extract_hrv_from_edf(uploaded_file, file_name):
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file_name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the EDF file
        edf_reader = pyedflib.EdfReader(temp_path)
        signal = edf_reader.readSignal(0)  # Read first channel (assumes ECG)
        sampling_rate = int(edf_reader.getSampleFrequency(0))
        edf_reader._close()

        # Preprocess ECG signal & detect R-peaks
        processed_ecg, ecg_info = nk.ecg_process(signal, sampling_rate=sampling_rate)
        r_peaks = ecg_info["ECG_R_Peaks"]

        # Extract HRV features
        hrv_features = nk.hrv(ecg_info, sampling_rate=sampling_rate, show=False)

        # Ensure all selected features are present
        missing_features = [feat for feat in selected_features if feat not in hrv_features.columns]
        if missing_features:
            st.warning(f"⚠️ Warning: Missing features in `{file_name}` → {missing_features}")
            return None, None, None

        # Select only important features
        hrv_selected = hrv_features[selected_features]

        return hrv_selected, signal, r_peaks

    except Exception as e:
        st.error(f"⚠️ Error processing `{file_name}`: {e}")
        return None, None, None

# 📌 Function to Assign Severity Levels Based on HRV Features
def assign_severity(hrv_features):
    lf_hf_ratio = hrv_features["HRV_LFHF"].values[0]
    
    if lf_hf_ratio < 1.5:
        return "Mild Risk"
    elif 1.5 <= lf_hf_ratio < 3:
        return "Moderate Risk"
    else:
        return "Severe Risk"

# 📌 Upload Multiple EDF Files
uploaded_files = st.file_uploader("📂 Upload Multiple ECG EDF Files", type=["edf"], accept_multiple_files=True)

if uploaded_files:
    st.write("✅ **Files uploaded successfully!** Processing...")

    results = []  # Store results for all files

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        st.write(f"🔍 Processing `{file_name}`...")

        # Extract HRV features, ECG signal, and R-peaks
        hrv_features, signal, r_peaks = extract_hrv_from_edf(uploaded_file, file_name)

        if hrv_features is not None:
            # Scale the HRV features
            hrv_scaled = pd.DataFrame(scaler.transform(hrv_features), columns=hrv_features.columns)

            # Make Prediction
            prediction = model.predict(hrv_scaled)[0]

            # 📌 Convert numerical label to meaningful diagnosis
            label_mapping = {0: "Normal ECG (No abnormality detected)", 1: "Abnormal ECG (Potential cardiac issue)"}
            diagnosis = label_mapping[prediction]

            # Assign severity level
            severity = assign_severity(hrv_features) if prediction == 1 else "None"

            # Store result
            results.append({"File Name": file_name, "Predicted Label": prediction, "Diagnosis": diagnosis, "Severity": severity})

            # 📌 Plot ECG with R-Peak Detection
            st.subheader(f"📊 ECG Waveform - `{file_name}`")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(signal[:5000], label="ECG Signal", alpha=0.8)
            ax.scatter(r_peaks[r_peaks < 5000], signal[r_peaks[r_peaks < 5000]], color='red', marker='o', label="R-Peaks")
            ax.legend()
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            plt.title("ECG Waveform with R-Peak Detection")
            st.pyplot(fig)

    # 📌 Display Prediction Results
    if results:
        st.subheader("✅ ECG Diagnosis Results")
        results_df = pd.DataFrame(results)
        st.write(results_df)

        # 📌 Download Results as CSV
        st.download_button(
            label="📥 Download Predictions",
            data=results_df.to_csv(index=False),
            file_name="predicted_ecg_results.csv",
            mime="text/csv"
        )
