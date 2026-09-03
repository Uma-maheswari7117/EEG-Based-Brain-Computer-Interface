
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt


# -----------------------------
# BANDPASS FILTER FUNCTION
# -----------------------------
def bandpass_filter(signal, lowcut, highcut, fs, order=4):

    nyquist = fs / 2

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    filtered_signal = filtfilt(
        b,
        a,
        signal
    )

    return filtered_signal


# -----------------------------
# READ EEG DATA
# -----------------------------
data = pd.read_csv("../data/eeg_data.csv")

eeg = data["EEG"].values


# -----------------------------
# REMOVE DC OFFSET
# -----------------------------
eeg = eeg - np.mean(eeg)


# -----------------------------
# FILTER SETTINGS
# -----------------------------
SAMPLING_RATE = 256

LOWCUT = 0.5
HIGHCUT = 30


# -----------------------------
# APPLY FILTER
# -----------------------------
filtered_eeg = bandpass_filter(
    eeg,
    LOWCUT,
    HIGHCUT,
    SAMPLING_RATE
)


# -----------------------------
# SAVE FILTERED DATA
# -----------------------------
output = pd.DataFrame({
    "EEG": filtered_eeg
})

output.to_csv(
    "../data/filtered_eeg.csv",
    index=False
)


# -----------------------------
# PLOT FILTERED EEG
# -----------------------------
time = np.arange(len(filtered_eeg)) / SAMPLING_RATE

plt.figure(figsize=(12, 5))

plt.plot(time, filtered_eeg)

plt.xlabel("Time (seconds)")
plt.ylabel("Filtered Amplitude")
plt.title("Filtered EEG Signal")

plt.grid(True)

plt.show()


print("Filtering completed!")
print("Filtered data saved as filtered_eeg.csv")