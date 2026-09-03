
import serial
import time
import csv
import os

PORT = "COM5"
BAUD_RATE = 115200

# CHANGE THESE VALUES
LABEL = 1
STATE_NAME = "focus"

DURATION = 120  # seconds


# Create data folder
data_folder = "../data"
os.makedirs(data_folder, exist_ok=True)

# File name
file_path = os.path.join(
    data_folder,
    f"{STATE_NAME}.csv"
)


# Connect Arduino
arduino = serial.Serial(
    PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)


print("--------------------------------")
print("EEG DATA COLLECTION STARTED")
print("STATE:", STATE_NAME.upper())
print("DURATION:", DURATION, "seconds")
print("--------------------------------")

print("\nGet ready...")

time.sleep(5)

start_time = time.time()


with open(
    file_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "EEG",
        "label"
    ])

    while time.time() - start_time < DURATION:

        line = arduino.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if line:

            try:

                eeg_value = int(line)

                timestamp = (
                    time.time()
                    - start_time
                )

                writer.writerow([
                    timestamp,
                    eeg_value,
                    LABEL
                ])

                print(
                    f"{timestamp:.2f}",
                    eeg_value
                )

            except ValueError:
                pass


arduino.close()

print("\nDATA COLLECTION COMPLETED!")

print("Saved to:")
print(file_path)