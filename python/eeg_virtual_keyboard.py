import tkinter as tk
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
# LOAD ML MODEL
# ==========================================

model = joblib.load("../model/eeg_model.pkl")

print("ML model loaded successfully!")


# ==========================================
# EEG FILTER
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

    return filtfilt(
        b,
        a,
        signal
    )


# ==========================================
# BAND POWER
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

    signal = signal - np.mean(signal)

    filtered = bandpass_filter(signal)

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
# VIRTUAL KEYBOARD
# ==========================================

root = tk.Tk()

root.title("EEG Based Virtual Keyboard")

root.geometry("1000x650")

root.configure(bg="white")


# ==========================================
# TITLE
# ==========================================

title = tk.Label(
    root,
    text="EEG-Based Brain Computer Interface",
    font=("Arial", 24, "bold"),
    bg="white"
)

title.pack(pady=15)


# ==========================================
# TEXT BOX
# ==========================================

text_box = tk.Entry(
    root,
    font=("Arial", 26),
    width=35,
    justify="center"
)

text_box.pack(pady=15)


# ==========================================
# STATUS
# ==========================================

status_label = tk.Label(
    root,
    text="Connecting to EEG...",
    font=("Arial", 18, "bold"),
    bg="white"
)

status_label.pack(pady=10)


# ==========================================
# KEYBOARD
# ==========================================

keyboard_frame = tk.Frame(
    root,
    bg="white"
)

keyboard_frame.pack(pady=15)


keys = [
    ["A", "B", "C", "D", "E", "F", "G"],
    ["H", "I", "J", "K", "L", "M", "N"],
    ["O", "P", "Q", "R", "S", "T", "U"],
    ["V", "W", "X", "Y", "Z", "SPACE", "CLEAR"]
]


buttons = []

selected_index = 0


# ==========================================
# KEY PRESS FUNCTION
# ==========================================

def select_key(key):

    if key == "SPACE":

        text_box.insert(
            tk.END,
            " "
        )

    elif key == "CLEAR":

        text_box.delete(
            0,
            tk.END
        )

    else:

        text_box.insert(
            tk.END,
            key
        )


# ==========================================
# CREATE BUTTONS
# ==========================================

for row_index, row in enumerate(keys):

    for column_index, key in enumerate(row):

        button = tk.Button(
            keyboard_frame,
            text=key,
            font=("Arial", 16, "bold"),
            width=8,
            height=2
        )

        button.grid(
            row=row_index,
            column=column_index,
            padx=5,
            pady=5
        )

        buttons.append(button)


# ==========================================
# HIGHLIGHT SELECTED KEY
# ==========================================

def update_selection():

    for i, button in enumerate(buttons):

        if i == selected_index:

            button.config(
                relief=tk.SUNKEN,
                bd=5
            )

        else:

            button.config(
                relief=tk.RAISED,
                bd=2
            )


# ==========================================
# MOVE TO NEXT KEY
# ==========================================

def move_next():

    global selected_index

    selected_index += 1

    if selected_index >= len(buttons):

        selected_index = 0

    update_selection()


# ==========================================
# SELECT CURRENT KEY
# ==========================================

def select_current():

    key = keys[
        selected_index // 7
    ][
        selected_index % 7
    ]

    select_key(key)


# ==========================================
# INITIAL SELECTION
# ==========================================

update_selection()


# ==========================================
# EEG CONNECTION
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

status_label.config(
    text="EEG Connected | Starting..."
)


# ==========================================
# EEG BUFFER
# ==========================================

eeg_buffer = []


# ==========================================
# PREDICTION HISTORY
# ==========================================

prediction_history = []

last_action = time.time()

ACTION_DELAY = 4


# ==========================================
# EEG PROCESSING
# ==========================================

def process_eeg():

    global eeg_buffer
    global prediction_history
    global last_action

    try:

        # Read multiple available EEG values
        for _ in range(20):

            if arduino.in_waiting:

                line = arduino.readline().decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                if line:

                    try:

                        eeg_value = int(line)

                        eeg_buffer.append(
                            eeg_value
                        )

                    except ValueError:

                        pass


        # Need 512 samples

        if len(eeg_buffer) >= WINDOW_SIZE:

            signal = np.array(
                eeg_buffer[-WINDOW_SIZE:]
            )

            # Remove processed samples
            eeg_buffer = []


            # Extract features

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


            # ML prediction

            prediction = model.predict(
                feature_data
            )[0]


            if prediction == 0:

                state = "RELAX"

            else:

                state = "FOCUS"


            # Store prediction

            prediction_history.append(
                state
            )


            if len(prediction_history) > 3:

                prediction_history.pop(0)


            # Stable prediction

            if len(prediction_history) == 3:

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


                status_label.config(
                    text=f"EEG: {stable_state}"
                )


                # ==================================
                # CONTROL KEYBOARD
                # ==================================

                current_time = time.time()


                if current_time - last_action >= ACTION_DELAY:

                    if stable_state == "FOCUS":

                        # FOCUS = move to next key

                        move_next()

                        print(
                            "FOCUS → Next key"
                        )

                        last_action = current_time


                    elif stable_state == "RELAX":

                        # RELAX = select key

                        select_current()

                        print(
                            "RELAX → Key selected"
                        )

                        last_action = current_time


    except Exception as e:

        print(
            "Error:",
            e
        )


    # Run again after 100 milliseconds

    root.after(
        100,
        process_eeg
    )


# ==========================================
# START EEG PROCESSING
# ==========================================

root.after(
    100,
    process_eeg
)


# ==========================================
# CLOSE FUNCTION
# ==========================================

def close_program():

    try:

        arduino.close()

    except:

        pass

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    close_program
)


# ==========================================
# START GUI
# ==========================================

root.mainloop()