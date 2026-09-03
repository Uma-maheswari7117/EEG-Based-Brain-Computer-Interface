import serial
import time
import numpy as np
import pandas as pd
import joblib

from scipy.signal import butter, filtfilt, welch


# ==========================================
# SETTINGS
# ==========================================

PORT = "COM5"
BAUD_RATE = 115200

SAMPLING_RATE = 256

WINDOW_SECONDS = 2
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS

LOWCUT = 0.5
HIGHCUT = 30


# ==========================================
# LOAD TRAINED ML MODEL
# ==========================================

model = joblib.load("../model/eeg_model.pkl")

print("ML model loaded successfully!")


# ==========================================
# BANDPASS FILTER
# ==========================================

def bandpass_filter(signal):

    nyquist = SAMPLING_RATE / 2

    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist

    b, a = butter(
        4,
        [low, high],
        btype="band"
    )

    filtered_signal = filtfilt(
        b,
        a,
        signal
    )

    return filtered_signal


# ==========================================
# CALCULATE EEG BAND POWER
# ==========================================

def band_power(signal, low_freq, high_freq):

    frequencies, power = welch(
        signal,
        fs=SAMPLING_RATE,
        nperseg=min(256, len(signal))
    )

    frequency_mask = (
        (frequencies >= low_freq) &
        (frequencies <= high_freq)
    )

    return np.trapezoid(
        power[frequency_mask],
        frequencies[frequency_mask]
    )


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(signal):

    # Remove DC component
    signal = signal - np.mean(signal)

    # Apply bandpass filter
    filtered = bandpass_filter(signal)

    # Delta: 0.5 - 4 Hz
    delta = band_power(
        filtered,
        0.5,
        4
    )

    # Theta: 4 - 8 Hz
    theta = band_power(
        filtered,
        4,
        8
    )

    # Alpha: 8 - 13 Hz
    alpha = band_power(
        filtered,
        8,
        13
    )

    # Beta: 13 - 30 Hz
    beta = band_power(
        filtered,
        13,
        30
    )

    # Mean
    mean = np.mean(filtered)

    # Standard deviation
    std = np.std(filtered)

    # IMPORTANT:
    # Keep the same feature order used during training

    return [
        delta,
        theta,
        alpha,
        beta,
        mean,
        std
    ]


# ==========================================
# CONNECT TO ARDUINO
# ==========================================

print()
print("Connecting to Arduino...")

arduino = serial.Serial(
    PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("Arduino connected!")

print()
print("Real-time EEG prediction started.")
print("----------------------------------------")


# ==========================================
# EEG BUFFER
# ==========================================

eeg_buffer = []


# ==========================================
# PREDICTION HISTORY
# ==========================================

prediction_history = []


# ==========================================
# REAL-TIME EEG LOOP
# ==========================================

try:

    while True:

        # Read data from Arduino
        line = arduino.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if line:

            try:

                # Convert received EEG value to integer
                eeg_value = int(line)

                # Add EEG value to buffer
                eeg_buffer.append(eeg_value)

                # Display collection progress
                print(
                    f"\rCollecting EEG samples: "
                    f"{len(eeg_buffer)}/{WINDOW_SIZE}",
                    end=""
                )

                # ======================================
                # WHEN 512 SAMPLES ARE COLLECTED
                # ======================================

                if len(eeg_buffer) >= WINDOW_SIZE:

                    print()

                    # Take latest 512 samples
                    signal = np.array(
                        eeg_buffer[-WINDOW_SIZE:]
                    )

                    # ==================================
                    # FEATURE EXTRACTION
                    # ==================================

                    features = extract_features(
                        signal
                    )

                    # ==================================
                    # CREATE FEATURE DATAFRAME
                    # ==================================

                    feature_data = pd.DataFrame(
                        [features],
                        columns=[
                            "delta",
                            "theta",
                            "alpha",
                            "beta",
                            "mean",
                            "std"
                        ]
                    )

                    # ==================================
                    # ML PREDICTION
                    # ==================================

                    prediction = model.predict(
                        feature_data
                    )[0]

                    # ==================================
                    # PREDICTION PROBABILITY
                    # ==================================

                    probabilities = model.predict_proba(
                        feature_data
                    )[0]

                    confidence = max(
                        probabilities
                    ) * 100

                    # ==================================
                    # CONVERT LABEL TO STATE
                    # ==================================

                    if prediction == 0:

                        state = "RELAX"

                    else:

                        state = "FOCUS"


                    # ==================================
                    # ADD TO PREDICTION HISTORY
                    # ==================================

                    prediction_history.append(
                        state
                    )


                    # Keep only latest 3 predictions

                    if len(prediction_history) > 3:

                        prediction_history.pop(0)


                    # ==================================
                    # DISPLAY CURRENT PREDICTION
                    # ==================================

                    print("----------------------------------------")

                    print(
                        "Current Prediction :",
                        state
                    )

                    print(
                        f"Confidence         : "
                        f"{confidence:.2f}%"
                    )

                    print(
                        "Last Predictions   :",
                        prediction_history
                    )

                    print("----------------------------------------")


                    # ==================================
                    # STABLE PREDICTION
                    # ==================================

                    if len(prediction_history) == 3:

                        relax_count = (
                            prediction_history.count(
                                "RELAX"
                            )
                        )

                        focus_count = (
                            prediction_history.count(
                                "FOCUS"
                            )
                        )


                        # Majority decision

                        if relax_count > focus_count:

                            final_state = "RELAX"

                        else:

                            final_state = "FOCUS"


                        print()
                        print(
                            "========================================"
                        )

                        print(
                            "STABLE PREDICTION :",
                            final_state
                        )

                        print(
                            "========================================"
                        )

                        print()


                    # ==================================
                    # CLEAR BUFFER
                    # ==================================

                    eeg_buffer = []


            except ValueError:

                # Ignore invalid Arduino data

                pass


# ==========================================
# STOP PROGRAM WITH CTRL + C
# ==========================================

except KeyboardInterrupt:

    print()
    print()
    print("Prediction stopped.")


# ==========================================
# CLOSE ARDUINO CONNECTION
# ==========================================

finally:

    arduino.close()

    print("Arduino connection closed.")