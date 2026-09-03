import serial
import time

PORT = "COM5"
BAUD_RATE = 115200

arduino = serial.Serial(PORT, BAUD_RATE, timeout=1)

time.sleep(2)

print("Connected to Arduino!")
print("Reading EEG data...\n")

try:
    while True:
        line = arduino.readline().decode("utf-8").strip()

        if line:
            print(line)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    arduino.close()


