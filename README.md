# 📌 WaveformNet: A 1D CNN-Based Model for Multiclass Arrhythmia Classification

**WaveformNet** is a deep learning model built using one-dimensional convolutional neural networks (1D CNNs) to classify ECG signals into multiple arrhythmia types. Designed for efficiency and accuracy, the model analyzes raw ECG waveforms to detect abnormal heart rhythms, supporting multiclass classification modes. It serves as a foundational step toward intelligent cardiac monitoring and AI-driven healthcare solutions.

---

## 📖 Table of Contents
- [Project Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset (if applicable)](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## 🔍 Project Description

**WaveformNet** is a 1D CNN-based deep learning model for classifying ECG signals into multiple arrhythmia types. It supports both multiclass classification (14 heartbeat categories) and binary classification to distinguish between normal and abnormal rhythms.

For binary classification, a simple logic is applied:

If the predicted class is 6 ('N'), the beat is labeled 'Normal'; otherwise, it's 'Abnormal'.

This project was developed as part of my AI/ML learning journey, aiming to gain hands-on experience in biomedical signal processing, deep learning model design, and real-world problem-solving. While not a production-ready system, it lays the groundwork for future projects in healthtech and machine learning.

It's intended for:

- Learners and developers exploring deep learning in healthcare
- Those interested in ECG signal classification
- Anyone looking for a practical example of CNNs applied to time-series biomedical data.

---

## ✨ Features

1. **Multiclass ECG Classification**: Classifies ECG signals into 14 distinct heartbeat types using a 1D CNN architecture.

2. **Binary Classification Mode**: Distinguishes between normal ('N') and abnormal beats using a simple ternary condition:
```python
'Normal' if idx == 6 else 'Abnormal'.
```

3. **End-to-End Deep Learning Pipeline**: Includes preprocessing, model training, evaluation, and inference.

4. **Educational Purpose**: Developed as a foundational project in a broader journey toward AI/ML engineering.
---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/project-name.git
cd project-name
pip install -r requirements.txt
```

## 📁 Project Structure

```tree
├── Annotation.csv
├── Encoded Classes.txt                         ## Original 23 classes including normal, non-beat and abnormal classes
├── Features.npy
├── History.csv                                 ## Training history data
├── Labels (Mutli Class).npy
├── Model                                       ## Saved model as tf format
│   ├── assets
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── Models                                      ## Models saved as .h5 and .keras formats
│   ├── Model.h5
│   └── Model.keras
├── Notebook PDFs                               ## Saved the Jupyter Notebooks as PDFs for future reference
│   ├── Data PP.pdf
│   ├── Inference.pdf
│   ├── Plots.pdf
│   └── Training (Multi Class).pdf
├── Notebooks
│   ├── Data PP.ipynb                           ## Data Preprocessing
│   ├── Inference.ipynb                         ## Inference on unseen data
│   ├── Plots.ipynb                             ## Evaluation and metrics
│   └── Training (Multi Class).ipynb            ## Training
├── Plots
│   ├── Accuracy Graphs.png
│   ├── Combined Graphs.png
│   ├── Confusion Matrix.png
│   └── Loss Graphs.png
├── README.md
├── Remapped_Symbol_Classes.txt
├── mitdb
│   ├── 100.atr
│   ├── 100.dat
│   ├── 100.hea
│   ├── 101.atr
│   ├── 101.dat
│   ├── 101.hea
│   ├── 102.atr
│   ├── 102.dat
│   ├── 102.hea	
│   ├── ...
│   ├── ...
│   ├── ...
```