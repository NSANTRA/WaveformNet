# 📌 WaveformNet: 1D and 2D CNN-Based Models for Multiclass Arrhythmia Classification

WaveformNet is a deep learning project for classifying electrocardiogram (ECG) signals into multiple arrhythmia types using convolutional neural networks. It features two separately trained models: a one-dimensional (1D) CNN that processes raw ECG time series, and a two-dimensional (2D) CNN that operates on transformed representations. Developed on the same dataset, the models enable a comparative study of temporal vs. spatiotemporal feature extraction for multiclass arrhythmia classification, advancing research in AI-powered cardiac diagnostics.

---

## 📖 Table of Contents
- [Project Description](#-project-description)
- [Features](#-features)
- [Installation & Prerequisites](#️-installation--prerequisites)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Results](#-results)

---

## 🔍 Project Description

**WaveformNet** is a deep learning project that includes two models: a 1D CNN-based model and a 2D CNN-based model, both designed to classify ECG signals into multiple arrhythmia types. The project supports multiclass classification (14 heartbeat categories) and binary classification to differentiate between normal and abnormal rhythms.

For binary classification, the logic is simple:

If the predicted class is 6 ('N'), the beat is labeled as 'Normal'; otherwise, it is labeled as 'Abnormal'.

Both models were trained on the same dataset:

- The 1D model processes raw ECG signals as time series data.
- The 2D model operates on transformed representations (e.g., spectrograms or scalograms), allowing for spatiotemporal feature extraction.

This project was developed as part of my AI/ML learning journey, providing hands-on experience in biomedical signal processing, deep learning model design, and real-world problem-solving. While not a production-ready system, it lays the groundwork for future healthtech and machine learning projects.

Intended for:
- Learners and developers interested in deep learning applications in healthcare.

- Researchers or practitioners focusing on ECG signal classification.

- Anyone seeking a practical example of CNNs applied to time-series biomedical data.

---

## ✨ Features

1. **Multiclass ECG Classification**: Classifies ECG signals into 14 distinct heartbeat types using a 1D CNN and a 2D CNN architecture.

2. **Binary Classification Mode**: Differentiates between normal ('N') and abnormal beats using a simple ternary condition:
```python
'Normal' if idx == 6 else 'Abnormal'.
```

3. **End-to-End Deep Learning Pipeline**: Includes preprocessing, model training, evaluation, and inference.

4. **Educational Purpose**: Developed as a foundational project in a broader AI/ML engineering journey to gain hands-on experience in biomedical signal processing.
---

## ⚙️ Installation & Prerequisites
### Prerequisites
Ensure you have the following installed:

- Python == 3.10.13
- pip == 24.2
- MIT-BIH Arrhythmia Dataset (can be downloaded via WFDB or manually)
- Git (optional for cloning)

Recommended Python Packages
```bash
pip install numpy pandas matplotlib seaborn scikit-learn wfdb tensorflow
```

Clone the Repository
```bash
git clone https://github.com/NSANTRA/WaveformNet-Arrhythmia-Classification.git
cd WaveformNet-Arrhythmia-Classification
```

📂 Dataset Setup
You can use the WFDB Python package to download the MIT-BIH dataset:

```python
import wfdb
wfdb.dl_database("mitdb", dl_dir = "mitdb")
```
Or download manually from [PhysioNet](#-dataset) and place it in a mitdb/ directory inside the project root.

---

## 📁 Project Structure

```tree
├── Annotation.csv                              ## Kind of metadata for annotation per patient ID
├── Encoded Classes.txt                         ## Original 23 classes including normal, non-beat and abnormal classes (symbol → label)
├── Features.npy                                ## Processed features
├── History 1D.csv                              ## 1D Model training history 
├── History 2D.csv                              ## 2D Model training history
├── Labels (Mutli Class).npy                    ## Processed labels (0 to 13)
├── Model 1D                                    ## Saved model (1D) as tf format for quantization purposes (future goal)
│   ├── assets
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── Model 2D                                    ## Saved model (2D) as tf format for quantization purposes (future goal)
│   ├── assets
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── Models                                      ## Models saved as .h5 (legacy) and .keras formats
│   ├── Model 1D.h5
│   ├── Model 1D.keras
│   ├── Model 2D.h5
│   └── Model 2D.keras
├── Notebook PDFs                               ## Saved the Jupyter Notebooks as PDFs for future reference
│   ├── 1D
│   │   ├── Inference 1D.pdf
│   │   ├── Plots 1D.pdf
│   │   └── Training (Multi Class) 1D.pdf
│   ├── 2D
│   │   ├── Inference 2D.pdf
│   │   ├── Plots 2D.pdf
│   │   └── Training (Multi Class) 2D.pdf
│   └── Data PP.pdf
├── Notebooks
│   ├── Data PP.ipynb                           ## Data Preprocessing
│   ├── Inference 1D.ipynb                      ## Inference on unseen data (1D)
│   ├── Inference 2D.ipynb                      ## Inference on unseen data (2D)
│   ├── Plots 1D.ipynb                          ## Evaluation and metrics (1D)
│   ├── Plots 2D.ipynb                          ## Evaluation and metrics (1D)
│   ├── Training (Multi Class) 1D.ipynb         ## Training (1D)
│   └── Training (Multi Class) 2D.ipynb         ## Training (2D)
├── Plots                                       ## Graphs plots saved
│   ├── 1D
│   │   ├── Accuracy Graphs 1D.png
│   │   ├── Combined Graphs 1D.png
│   │   ├── Confusion Matrix 1D.png
│   │   └── Loss Graphs 1D.png
│   └── 2D
│       ├── Accuracy Graphs 2D.png
│       ├── Combined Graphs 2D.png
│       ├── Confusion Matrix 2D.png
│       └── Loss Graphs 2D.png
├── README.md
├── Remapped_Symbol_Classes.txt
├── mitdb                                       ## Main dataset
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

---

## 🧬 Dataset

### MIT-BIH Arrhythmia Database (MITDB)
The **MIT-BIH Arrhythmia Database** is a widely used benchmark dataset in biomedical signal processing, particularly for developing and evaluating algorithms for automated ECG (electrocardiogram) analysis. It consists of 48 half-hour excerpts of two-channel ambulatory ECG recordings, collected from 47 subjects by the Beth Israel Hospital Arrhythmia Laboratory between 1975 and 1979. Each record includes manually annotated beat and rhythm labels verified by independent experts.

MITDB serves as a gold standard for tasks such as heartbeat classification, arrhythmia detection, and ECG signal segmentation.

- **Format:** PhysioBank-compatible .dat, .hea, and .atr files.
- **Sampling Frequency:** 360 Hz
- **Annotations:** Beat types and rhythms using AAMI EC57 standard labels

Official Source:
MIT-BIH Arrhythmia Database — [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

---

## 📈 Results

### Training & Validation Metrics (1D)
The model was trained for 50 epochs on the MIT-BIH Arrhythmia Dataset. The following plots demonstrate the model's performance:

#### Training vs Validation Loss
- The training and validation loss curves steadily decrease and converge, indicating proper learning and no signs of overfitting. Final validation loss stabilizes near zero.

![Loss Graph](Plots/1D/Loss%20Graphs%201D.png)

#### Training vs Validation Accuracy
- The model achieves over 98% validation accuracy, demonstrating strong generalization capability.
- Accuracy plateaued after ~30 epochs, suggesting optimal convergence.

![Accuracy Graph](Plots/1D/Accuracy%20Graphs%201D.png)

#### Combined Accuracy & Loss Overview
- This side-by-side visualization offers a comprehensive look at the tradeoff between accuracy and loss. Both metrics indicate consistent improvement during training.

![Combined Graph](Plots/1D/Combined%20Graphs%201D.png)

#### Confusion Matrix
- The confusion matrix shows strong classification performance across most classes. Diagonal dominance indicates accurate predictions.
- Some minor misclassifications are present in adjacent classes, which is common in ECG signal tasks.

![Confusion Matrix](Plots/1D/Confusion%20Matrix%201D.png)

***

### Training & Validation Metrics (2D)
The models were trained for 50 epochs on the MIT-BIH Arrhythmia Dataset, and the performance metrics reflect strong generalization and learning behavior.

#### Training vs Validation Loss
- The loss curves for both training and validation datasets indicate smooth and effective convergence.
- Training loss steadily decreases and approaches zero.
- Validation loss remains consistently low throughout training, with no major spikes — a strong indicator of minimal overfitting.

The model demonstrates excellent optimization stability.

![Loss Graph](Plots/2D/Loss%20Graphs%202D.png)

#### Training vs Validation Accuracy
Accuracy trends confirm robust learning:
- Training accuracy reaches ~99.7%, and validation accuracy maintains above 98.9%.
- Both curves plateau after around 30 epochs, indicating early convergence and model generalization.
- The narrow gap between training and validation accuracy suggests balanced performance without overfitting.

![Accuracy Graph](Plots/2D/Accuracy%20Graphs%202D.png)

#### Combined Accuracy & Loss Overview
This dual-pane visualization presents a clear overview:
- Consistent improvement in accuracy across epochs.
- Parallel reduction in loss values, reflecting strong correlation between optimization and classification performance.
- Highlights the model’s ability to learn complex ECG patterns efficiently.

![Combined Graph](Plots/2D/Combined%20Graphs%202D.png)

#### Confusion Matrix
The confusion matrix further supports high performance:
- Strong diagonal dominance indicates high precision and recall across most classes.
- Minor misclassifications appear primarily between adjacent or morphologically similar heartbeat types — an expected challenge in ECG signal classification.
- Overall class-wise predictions are highly reliable, even in less represented categories.

![Confusion Matrix](Plots/2D/Confusion%20Matrix%202D.png)

---