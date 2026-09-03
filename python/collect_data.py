import serial
import time
import csv
import os

PORT = "COM5"
BAUD_RATE = 115200

# Location where the CSV file will be saved
data_folder = "../data"
os.makedirs(data_folder, exist_ok=True)

file_path = os.path.join(data_folder, "eeg_data.csv")

# Connect to Arduino
arduino = serial.Serial(PORT, BAUD_RATE, timeout=1)

time.sleep(2)

print("Connected to Arduino!")
print("Collecting EEG data...")
print("Press Ctrl+C to stop.\n")

with open(file_path, "w", newline="") as file:

    writer = csv.writer(file)

    # CSV column name
    writer.writerow(["EEG"])

    try:
        while True:

            line = arduino.readline().decode("utf-8").strip()

            if line:

                try:
                    eeg_value = int(line)

                    writer.writerow([eeg_value])

                    print(eeg_value)

                except ValueError:
                    pass

    except KeyboardInterrupt:

        print("\nData collection stopped.")

arduino.close()

print("EEG data saved to:")
print(file_path)
