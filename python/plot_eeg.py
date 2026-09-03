import pandas as pd
import matplotlib.pyplot as plt

# Read EEG data
data = pd.read_csv("../data/eeg_data.csv")

# Get EEG values
eeg = data["EEG"]

# Create time axis
sampling_rate = 256
time = [i / sampling_rate for i in range(len(eeg))]

# Plot EEG signal
plt.figure(figsize=(12, 5))

plt.plot(time, eeg)

plt.xlabel("Time (seconds)")
plt.ylabel("EEG Amplitude")
plt.title("Raw EEG Signal")

plt.grid(True)

plt.show()