# EEG-Based Brain–Computer Interface

## Overview

This project implements an **EEG-based Brain–Computer Interface (BCI)** that converts brain signals into computer commands using machine learning.

EEG signals are captured using **EEG electrodes, BioAmp EXG Pill, and Arduino Uno**. The signals are processed in Python, features are extracted, and a **Random Forest classifier** predicts the user's mental state as **Relax or Focus**.

The predictions are used to control a **virtual keyboard** and **PowerPoint presentations** without physically using a keyboard.

## Project Pipeline

```text
EEG Electrodes
      ↓
BioAmp EXG Pill
      ↓
Arduino Uno
      ↓
Python
      ↓
EEG Preprocessing
      ↓
Feature Extraction
      ↓
Random Forest
      ↓
Relax / Focus
      ↓
Virtual Keyboard / PowerPoint
```

## Hardware

* Arduino Uno
* BioAmp EXG Pill
* BioAmp Cable V3
* 3 EEG Electrodes
* Jumper Wires
* USB Cable

## Technologies

* Python
* Arduino
* NumPy
* Pandas
* SciPy
* Scikit-learn
* PySerial
* Joblib
* Tkinter
* PyAutoGUI

## Features

* EEG signal acquisition
* EEG signal preprocessing
* Delta, Theta, Alpha and Beta feature extraction
* Relax/Focus classification
* Real-time EEG prediction
* Virtual keyboard control
* PowerPoint slide control

## Machine Learning

**Algorithm:** Random Forest Classifier

**Input Features:**

* Delta
* Theta
* Alpha
* Beta
* Mean
* Standard Deviation

**Classes:**

* `0` → Relax
* `1` → Focus

Initial model accuracy: **68%**

## Project Structure

```text
EEG_BCI_Project/
├── python/
│   ├── test_serial.py
│   ├── collect_data.py
│   ├── plot_eeg.py
│   ├── preprocess.py
│   ├── collect_labeled_data.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── realtime_prediction.py
│   ├── virtual_keyboard.py
│   ├── eeg_virtual_keyboard.py
│   └── eeg_powerpoint_control.py
│
├── data/
├── model/
└── README.md
```

## Future Improvements

* Collect more EEG training data
* Improve prediction accuracy
* Add advanced machine learning/deep learning models
* Improve real-time stability
* Add wireless IoT communication
* Add cloud integration

