import os
import wfdb
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Input
path = input("Enter the absolute path to the directory containing the WFDB files: ")
pid = input("Enter the patient ID: ")

# Load signal and annotations
rec = wfdb.rdrecord(os.path.join(path, pid))
signal = rec.p_signal
ann = wfdb.rdann(os.path.join(path, pid), "atr")

r_peaks_indices = ann.sample
symbols = ann.symbol

# Slice signal around R-peaks
X, y = [], []
for idx, r_peak in enumerate(r_peaks_indices):
    start, end = r_peak - 100, r_peak + 150
    if start >= 0 and end < signal.shape[0]:
        X.append(signal[start:end, :].astype(np.float16))
        y.append(symbols[idx])

X = np.array(X)
y = np.array(y)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y).astype(np.int8)

# Load symbol to code mapping
symbol, code = [], []
with open("Remapped_Symbol_Classes.txt", "r", encoding="utf-8") as file:
    for line in file:
        if "→" in line:
            s, c = line.strip().split("→")
            symbol.append(s.rstrip())
            code.append(c.lstrip())

# Load model
model = tf.keras.models.load_model("./Models/Model 1D.h5")

# Predict
for i in range(X.shape[0]):
    temp = np.expand_dims(X[i], axis=0)
    pred_idx = np.argmax(model.predict(temp, verbose=0))
    pred_symbol = encoder.classes_[pred_idx]
    if pred_symbol in symbol:
        mapped_index = symbol.index(pred_symbol)
        print(f"Predicted: {symbol[mapped_index]} ({code[mapped_index]}) for R-peak at index {r_peaks_indices[i]}")
    else:
        print(f"Predicted: {pred_symbol} (Unmapped) for R-peak at index {r_peaks_indices[i]}")