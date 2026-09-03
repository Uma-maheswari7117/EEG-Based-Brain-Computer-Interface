
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
import os


# ==============================
# SETTINGS
# ==============================

SAMPLING_RATE = 256

WINDOW_SECONDS = 2

WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS

LOWCUT = 0.5
HIGHCUT = 30


# ==============================
# BANDPASS FILTER
# ==============================

def bandpass_filter(signal):

    nyquist = SAMPLING_RATE / 2

    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist

    b, a = butter(
        4,
        [low, high],
        btype="band"
    )

    return filtfilt(b, a, signal)


# ==============================
# BAND POWER
# ==============================

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


# ==============================
# EXTRACT FEATURES
# ==============================

def extract_features(window):

    filtered = bandpass_filter(window)

    delta = band_power(filtered, 0.5, 4)

    theta = band_power(filtered, 4, 8)

    alpha = band_power(filtered, 8, 13)

    beta = band_power(filtered, 13, 30)

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


# ==============================
# PROCESS ONE FILE
# ==============================

def process_file(filename):

    data = pd.read_csv(filename)

    eeg = data["EEG"].values

    label = data["label"].iloc[0]

    features = []

    number_of_windows = len(eeg) // WINDOW_SIZE

    print(
        filename,
        "→",
        number_of_windows,
        "windows"
    )

    for i in range(number_of_windows):

        start = i * WINDOW_SIZE

        end = start + WINDOW_SIZE

        window = eeg[start:end]

        feature_values = extract_features(window)

        features.append(
            feature_values + [label]
        )

    return features


# ==============================
# MAIN
# ==============================

relax_file = "../data/relax.csv"

focus_file = "../data/focus.csv"


all_features = []


# Relax
all_features.extend(
    process_file(relax_file)
)


# Focus
all_features.extend(
    process_file(focus_file)
)


# ==============================
# CREATE DATAFRAME
# ==============================

columns = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "mean",
    "std",
    "label"
]


feature_data = pd.DataFrame(
    all_features,
    columns=columns
)


# ==============================
# SAVE
# ==============================

output_file = "../data/features.csv"

feature_data.to_csv(
    output_file,
    index=False
)


print("\nFeature extraction completed!")

print(
    "Saved to:",
    output_file
)

print(
    "\nTotal training samples:",
    len(feature_data)
)

print(
    "\nClass distribution:"
)

print(
    feature_data["label"].value_counts()
)