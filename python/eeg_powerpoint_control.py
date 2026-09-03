
import serial
import time
import numpy as np
import pandas as pd
import joblib
import pyautogui

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

# Number of predictions used for stable decision
STABLE_COUNT = 3

# Time between PowerPoint commands
ACTION_DELAY = 5


# ==========================================
# LOAD ML MODEL
# ==========================================

model = joblib.load("../model/eeg_model.pkl")

print("ML model loaded successfully!")


# ==========================================
# EEG BANDPASS FILTER
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
# EEG BAND POWER
# ==========================================

def band_power(signal, low_freq, high_freq):

    frequencies, power = welch(
        signal,
        fs=SAMPLING_RATE,
        nperseg=min(256, len(signal))
    )

    mask = (
        (frequencies >= low_freq) &
        (frequencies <= high_freq)
    )

    return np.trapezoid(
        power[mask],
        frequencies[mask]
    )


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(signal):

    # Remove DC offset
    signal = signal - np.mean(signal)

    # Filter EEG
    filtered = bandpass_filter(signal)

    # EEG frequency bands
    delta = band_power(
        filtered,
        0.5,
        4
    )

    theta = band_power(
        filtered,
        4,
        8
    )

    alpha = band_power(
        filtered,
        8,
        13
    )

    beta = band_power(
        filtered,
        13,
        30
    )

    # Statistical features
    mean = np.mean(filtered)
    std = np.std(filtered)

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


# ==========================================
# PREPARE REAL-TIME EEG
# ==========================================

eeg_buffer = []

prediction_history = []

last_action_time = 0

last_command = None


# ==========================================
# START POWERPOINT CONTROL
# ==========================================

print()
print("==========================================")
print(" EEG POWERPOINT CONTROL")
print("==========================================")

print()
print("CONTROL:")
print("FOCUS → Next Slide")
print("RELAX → Previous Slide")

print()
print("IMPORTANT:")
print("1. Open PowerPoint")
print("2. Start Slide Show")
print("3. Click the slideshow window")
print("4. Do not click another application")
print()

input("Press ENTER to start EEG control...")

print()
print("EEG PowerPoint control started!")
print("Press CTRL+C to stop.")
print()


# ==========================================
# MAIN LOOP
# ==========================================

try:

    while True:

        # ----------------------------------
        # READ EEG DATA
        # ----------------------------------

        line = arduino.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if line:

            try:

                eeg_value = int(line)

                eeg_buffer.append(eeg_value)

            except ValueError:

                pass


        # ----------------------------------
        # CHECK WHETHER WINDOW IS COMPLETE
        # ----------------------------------

        if len(eeg_buffer) >= WINDOW_SIZE:

            signal = np.array(
                eeg_buffer[-WINDOW_SIZE:]
            )

            eeg_buffer = []


            # ----------------------------------
            # EXTRACT FEATURES
            # ----------------------------------

            features = extract_features(
                signal
            )


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


            # ----------------------------------
            # ML PREDICTION
            # ----------------------------------

            prediction = model.predict(
                feature_data
            )[0]


            probabilities = model.predict_proba(
                feature_data
            )[0]


            confidence = max(
                probabilities
            ) * 100


            # ----------------------------------
            # CONVERT PREDICTION
            # ----------------------------------

            if prediction == 0:

                state = "RELAX"

            else:

                state = "FOCUS"


            print(
                f"Prediction: {state} | "
                f"Confidence: {confidence:.2f}%"
            )


            # ----------------------------------
            # STORE PREDICTION
            # ----------------------------------

            prediction_history.append(
                state
            )


            if len(prediction_history) > STABLE_COUNT:

                prediction_history.pop(0)


            # ----------------------------------
            # STABLE PREDICTION
            # ----------------------------------

            if len(prediction_history) == STABLE_COUNT:

                relax_count = prediction_history.count(
                    "RELAX"
                )

                focus_count = prediction_history.count(
                    "FOCUS"
                )


                if relax_count > focus_count:

                    stable_state = "RELAX"

                else:

                    stable_state = "FOCUS"


                print(
                    "Stable Prediction:",
                    stable_state
                )


                # ----------------------------------
                # CHECK ACTION DELAY
                # ----------------------------------

                current_time = time.time()


                if (
                    current_time - last_action_time
                    >= ACTION_DELAY
                ):


                    # ----------------------------------
                    # FOCUS → NEXT SLIDE
                    # ----------------------------------

                    if stable_state == "FOCUS":

                        pyautogui.press(
                            "right"
                        )

                        print(
                            ">>> NEXT SLIDE"
                        )

                        last_action_time = (
                            current_time
                        )

                        last_command = "FOCUS"


                    # ----------------------------------
                    # RELAX → PREVIOUS SLIDE
                    # ----------------------------------

                    elif stable_state == "RELAX":

                        pyautogui.press(
                            "left"
                        )

                        print(
                            "<<< PREVIOUS SLIDE"
                        )

                        last_action_time = (
                            current_time
                        )

                        last_command = "RELAX"


                print("------------------------------------------")


# ==========================================
# STOP PROGRAM
# ==========================================

except KeyboardInterrupt:

    print()
    print("PowerPoint control stopped.")


finally:

    arduino.close()

    print("Arduino connection closed.")