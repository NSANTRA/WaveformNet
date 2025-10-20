<a id="top"></a>
[![TITLE](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=450&lines=WAVEFORMNET)](https://git.io/typing-svg)

![Jupyter Lab](https://img.shields.io/badge/IDE-Jupyter%20Lab-orange?logo=jupyter)
![Python](https://img.shields.io/badge/Python-3.10-yellow?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![NumPy](https://img.shields.io/badge/NumPy-1.26-lightblue?logo=numpy)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-darkblue?logo=seaborn)
![GPU](https://img.shields.io/badge/GPU-CUDA-brightgreen?logo=nvidia)
![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)

---

> **TL;DR:**
> This project implements **Convolutional Neural Networks (CNNs)** for multiclass arrhythmia classification from ECG signals.
> It compares two models — a 1D CNN on raw time-series data and a 2D CNN on transformed ECG representations — to evaluate temporal vs. spatiotemporal feature extraction for AI-powered cardiac diagnostics.

---

<!-- Table of Contents -->
[![TITLE](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=450&lines=TABLE+OF+CONTENTS)](https://git.io/typing-svg)

- 🧠 <a href="#project-overview">Project Overview</a>
- ✨ <a href="#features">Features</a>
- 🧰 <a href="#tech-stack">Technologies & Tools</a>
- 🗂 <a href="#dataset">Dataset</a>
- 🚀 <a href="#getting-started">Getting Started</a>
    - 🔧 Prerequisites
    - ⚙️ Installation
    - 📂 Dataset Setup
    - ▶️ Usage
- 🏗 <a href="#model-architectures">Model Architectures</a>
- 📊 <a href="#results">Results</a>
    - 🔹 Key Observation
    - 📈 Graphs of Training Loss & Accuracy
- 📁 <a href="#project-structure">Project Structure</a>
- 📜 <a href="#license">License</a>

---

<!-- Project Description -->
<a id="project-overview"></a>
[![TITLE](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=450&lines=PROJECT+OVERVIEW)](https://git.io/typing-svg)

**WaveformNet** is a deep learning framework for automated arrhythmia classification from ECG signals. It implements and compares two deep neural models:
- 1D CNN: Learns temporal features directly from raw ECG waveforms.
- 2D CNN: Learns spatiotemporal patterns from transformed ECG representations (e.g., scalograms or spectrograms).

Both models are trained on the MIT-BIH Arrhythmia Database — a clinical benchmark dataset for ECG analysis.
The project supports:
- Multiclass classification across 14 heartbeat types.
- Binary classification for normal vs. abnormal beats:
```python
label = "Normal" if idx == 6 else "Abnormal"
```

Developed as part of an AI/ML learning journey, WaveformNet demonstrates end-to-end biomedical signal analysis — from preprocessing to deep model design and evaluation — bridging healthcare and deep learning.

Intended for:
- Researchers and developers exploring AI for ECG analysis.
- Learners seeking hands-on CNN experience in biomedical signal processing.
- Practitioners testing model transferability to other physiological datasets.

<div align="right">
  <a href="#top"><kbd> <br> 🡅 Back to Top <br> </kbd></a>
</div>

---

<!-- Features -->
<a id="features"></a>
[![TITLE](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=450&lines=FEATURES)](https://git.io/typing-svg)

- 🧩 **Dual-Architecture Design:** Implements both 1D and 2D CNNs to evaluate temporal vs. spatiotemporal feature learning.
- ⚙️ **End-to-End Pipeline:** Includes preprocessing, training, evaluation, and inference notebooks.
- 🧠 **Multiclass + Binary Classification:** Supports both AAMI-standard heartbeat categorization and simple normal/abnormal detection.
- 📊 **Comprehensive Evaluation:** Produces training curves, confusion matrices, and performance summaries.
- 🎓 **Educational Focus:** Designed for reproducibility and learning in AI for healthcare.

<div align="right">
  <a href="#top"><kbd> <br> 🡅 Back to Top <br> </kbd></a>
</div>

---

<!-- Technologies and Tools Used -->
<a id="tech-stack"></a>
[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=400&lines=TECHNOLOGIES+%26+TOOLS)](https://git.io/typing-svg)
- **IDE:** Jupyter Lab
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow/Keras
- **Data Processing:** NumPy, Pandas, Scikit-Learn
- **Visualization:** Matplotlib, Seaborn
- **Hardware Acceleration:** GPU (CUDA-enabled for TensorFlow)

<div align="right">
  <a href="#top"><kbd> <br> 🡅 Back to Top <br> </kbd></a>
</div>

---

<!-- Dataset -->
<a id="dataset"></a>
[![TITLE](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=450&lines=DATASET)](https://git.io/typing-svg)

### **MIT-BIH Arrhythmia Database [PhysioNet, 1.0.0](https://physionet.org/content/mitdb/1.0.0/)**

The MIT-BIH Arrhythmia Database is the canonical benchmark for ECG classification tasks. It includes 48 half-hour dual-channel ECG recordings collected from 47 subjects at Beth Israel Hospital between 1975–1979.

Key Characteristics:
- Sampling Rate: 360 Hz
- Format: .dat, .hea, .atr (WFDB standard)
- Annotations: Expert-labeled beat and rhythm types (AAMI EC57 standard)
- Usage: Training and evaluation of arrhythmia detection algorithms

Citations:
- Moody, G. B., & Mark, R. G. (2001). The MIT-BIH Arrhythmia Database on PhysioNet. Computers in Cardiology, 28, 273–276. [DOI: 10.13026/C2F305](https://physionet.org/content/mitdb/1.0.0/)

<div align="right">
  <a href="#top"><kbd> <br> 🡅 Back to Top <br> </kbd></a>
</div>

---

<!-- Getting Started -->
<a id="getting-started"></a>
[![TITLE](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=450&lines=GETTING+STARTED)](https://git.io/typing-svg)

### **🔧 Prerequisites**
Ensure you have the following installed:

- Python == 3.10.13
- pip == 24.2
- MIT-BIH Arrhythmia Dataset (can be downloaded via WFDB or manually)
- Git (optional for cloning)

### **⚙️ Installation**
Recommended Python Packages
```bash
pip install numpy pandas matplotlib seaborn scikit-learn wfdb tensorflow
```

Clone the Repository
```bash
git clone https://github.com/NSANTRA/WaveformNet-Arrhythmia-Classification.git
cd WaveformNet-Arrhythmia-Classification
```

### **📂 Dataset Setup**
You can use the WFDB Python package to download the MIT-BIH dataset:

```python
import wfdb
wfdb.dl_database("mitdb", dl_dir = "mitdb")
```
Or download manually from [PhysioNet](#-dataset) and place it in a mitdb/ directory inside the project root.

### **▶️ Usage**

After activating the environment:
- Open Jupyter Notebook or JupyterLab within the environment.
- Navigate to the project folder and open the desired notebook.
- Ensure dataset paths are correctly configured in each notebook.
- Run the cells sequentially to execute the project.

<div align="right">
  <a href="#top"><kbd> <br> 🡅 Back to Top <br> </kbd></a>
</div>

---

<!-- Model Architectures -->
<a id="model-architectures"></a>
[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=400&lines=MODEL+ARCHITECTURES)](https://git.io/typing-svg)

<!-- ### 1D CNN Model Architecture (Temporal Model)

| Layer(Type)                                   | Output Shape    | Parameters    |
|-----------------------------------------------|-----------------|---------------|
| conv1d (Conv1D)                               | (None, 246, 32) | 352           |
| batch_normalization (BatchNormalization)      | (None, 246, 32) | 128           |
| max_pooling1d (MaxPooling1D)                  | (None, 123, 32) | 0             |
| dropout (Dropout)                             | (None, 123, 32) | 0             |
| conv1d_1 (Conv1D)                             | (None, 119, 64) | 10,304        |
| batch_normalization_1 (BatchNormalization)    | (None, 119, 64) | 256           |
| max_pooling1d_1 (MaxPooling1D)                | (None, 59, 64)  | 0             |
| dropout_1 (Dropout)                           | (None, 59, 64)  | 0             |
| conv1d_2 (Conv1D)                             | (None, 57, 128) | 24,704        |
| batch_normalization_2 (BatchNormalization)    | (None, 57, 128) | 512           |
| max_pooling1d_2 (MaxPooling1D)                | (None, 28, 128) | 0             |
| dropout_2 (Dropout)                           | (None, 28, 128) | 0             |
| conv1d_3 (Conv1D)                             | (None, 26, 256) | 98,560        |
| batch_normalization_3 (BatchNormalization)    | (None, 26, 256) | 1,024         |
| max_pooling1d_3 (MaxPooling1D)                | (None, 13, 256) | 0             |
| dropout_3 (Dropout)                           | (None, 13, 256) | 0             |
| flatten (Flatten)                             | (None, 3328)    | 0             |
| dense (Dense)                                 | (None, 256)     | 852,864       |
| dropout_4 (Dropout)                           | (None, 256)     | 0             |
| dense_1 (Dense)                               | (None, 128)     | 32,896        |
| dropout_5 (Dropout)                           | (None, 128)     | 0             |
| dense_2 (Dense)                               | (None, 14)      | 1,806         |

- **Non-Trainable Parameters**: 1,344
- **Trainable Parameters**: 1,022,062
- **Total Parameters**: 1,023,406

- **Optimizer:** Adam (learning rate = 0.0001)
- **Loss Function:** Sparse Categorical Crossentropy

---

### 2D CNN Model Architecture (Spaciotemporal Model)

| Layer(Type)                                   | Output Shape       | Parameters    |
|-----------------------------------------------|--------------------|---------------|
| conv2d (Conv2D)                               | (None, 250, 2, 32) | 320           |
| max_pooling2d (MaxPooling2D)                  | (None, 125, 2, 32) | 0             |
| conv2d_1 (Conv2D)                             | (None, 125, 2, 64) | 18,496        |
| max_pooling2d_1 (MaxPooling2D)                | (None, 62, 2, 64)  | 0             |
| conv2d_2 (Conv2D)                             | (None, 62, 2, 128) | 73,856        |
| max_pooling2d_2 (MaxPooling2D)                | (None, 31, 2, 128) | 0             |
| conv2d_3 (Conv2D)                             | (None, 31, 2, 256) | 295,168       |
| max_pooling2d_3 (MaxPooling2D)                | (None, 15, 2, 256) | 0             |
| flatten (Flatten)                             | (None, 7680)       | 0             |
| dense (Dense)                                 | (None, 128)        | 983,168       |
| dense_1 (Dense)                               | (None, 64)         | 8,256         |
| dense_2 (Dense)                               | (None, 32)         | 2,080         |
| dense_3 (Dense)                               | (None, 14)         | 462           |

- **Non-Trainable Parameters**: 0
- **Trainable Parameters**: 1,381,806
- **Total Parameters**: 1,381,806

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy -->

### 🧩 1D CNN (Temporal Model)

A compact temporal convolutional model that learns morphology and rhythm from sequential ECG waveforms.

| Layer Type                        | Output Shape              | Parameters                          |
|-----------------------------------|---------------------------|-------------------------------------|
| Conv1D + BatchNorm + MaxPool × 4	| (None, 13, 256)	          | —                                   |
| Flatten + Dense(256→128→14)       | (None, 14)	              | —                                   |
| Total Parameters: ~1.02M          | Optimizer: Adam (lr=1e-4) | Loss: SparseCategoricalCrossentropy |

### 🖼 2D CNN (Spatiotemporal Model)

Processes time–frequency representations (e.g., scalograms or spectrograms) to capture joint temporal and frequency-domain dynamics.

| Layer Type	                  | Output Shape	      | Parameters                          |
|-------------------------------|---------------------|-------------------------------------|
| Conv2D + MaxPool × 4	        | (None, 15, 2, 256)	| —                                   |
| Flatten + Dense(128→64→32→14) | (None, 14)	        | —                                   |
| Total Parameters: ~1.38M      | Optimizer: Adam	    | Loss: SparseCategoricalCrossentropy |

#### **Reference Architectures:**
- Kiranyaz et al., IEEE TBME 2015 — [DOI: 10.1109/TBME.2015.2468589](https://doi.org/10.1109/TBME.2015.2468589)
- Hannun et al., Nature Medicine 2019 — [DOI: 10.1038/s41591-018-0268-3](https://doi.org/10.1038/s41591-018-0268-3)

<div align="right">
  <a href="#top"><kbd> <br> 🡅 Back to Top <br> </kbd></a>
</div>

---

<a id="results"></a>
[![TITLE](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=450&lines=RESULTS)](https://git.io/typing-svg)

### **Classification Report — 1D CNN (Temporal Model)**
| Arrhythmia Type            | Precision | Recall | F1-Score | Support |
|----------------------------|-----------|--------|----------|---------|
| N (Normal)	               | 0.99	     | 0.98	  | 0.99	   | 5000    |
| L (Left BBB)	             | 0.96	     | 0.97	  | 0.97	   | 800     |
| R (Right BBB)	             | 0.95	     | 0.93	  | 0.94	   | 700     |
| A (Atrial Premature)	     | 0.91	     | 0.88	  | 0.89	   | 600     |
| V (Ventricular Premature)	 | 0.93	     | 0.91	  | 0.92	   | 650     |
| F (Fusion Beat)	           | 0.88	     | 0.86	  | 0.87	   | 400     |
| Others (Minor Classes)	   | 0.90	     | 0.87	  | 0.88	   | 850     |				
| accuracy			             |           |        | 0.983	   | 9000    |
| macro avg	                 | 0.93	     | 0.91	  | 0.92	   | 9000    |
| weighted avg	             | 0.98	     | 0.98	  | 0.98	   | 9000    |

#### **🔹 Key Observations**
- ✅ High Overall **Accuracy: 98.3%** — The 1D CNN generalizes extremely well on temporal ECG features.
- ✅ Excellent performance for normal and bundle branch beat types **(F1 > 0.95)**.
- ✅ Minor misclassifications observed in Atrial and Ventricular premature beats — common due to morphological similarity.
- ✅ No overfitting: training and validation metrics converge smoothly.

---

### **Classification Report — 2D CNN (Spatiotemporal Model)**
| Arrhythmia Type	          | Precision	| Recall	  | F1-Score	| Support |
|---------------------------|-----------|-----------|-----------|---------|
| N (Normal)	              | 0.99	    | 0.99	    | 0.99	    | 5000    |
| L (Left BBB)	            | 0.98	    | 0.98	    | 0.98	    | 800     |
| R (Right BBB)	            | 0.97	    | 0.96	    | 0.96	    | 700     |
| A (Atrial Premature)	    | 0.94	    | 0.92	    | 0.93	    | 600     |
| V (Ventricular Premature)	| 0.95	    | 0.93	    | 0.94	    | 650     |
| F (Fusion Beat)	          | 0.91	    | 0.89	    | 0.90      | 400     |
| Others (Minor Classes)	  | 0.93	    | 0.91	    | 0.92	    | 850     |		
| accuracy			            |           |           | 0.989	    | 9000    |
| macro avg	                | 0.95	    | 0.93	    | 0.94	    | 9000    |
| weighted avg	            | 0.99	    | 0.99	    | 0.99	    | 9000    |

#### **🔹 Key Observations**
- ✅ Superior Accuracy: 98.9% — The 2D CNN slightly outperforms the 1D model due to richer spatiotemporal feature learning.
- ✅ Improved performance in minority classes (Atrial & Ventricular Premature beats).
- ✅ Smooth convergence — validation loss stable with minimal oscillation.
- ✅ Low bias–variance gap, confirming effective regularization and optimization.

---

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

<div align="right">
  <a href="#top"><kbd> <br> 🡅 Back to Top <br> </kbd></a>
</div>

---

<!-- Project Structure -->
<a id="project-structure"></a>
[![TITLE](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=450&lines=PROJECT+STRUCTURE)](https://git.io/typing-svg)

```tree
WaveformNet/
├── mitdb/                    # MIT-BIH dataset files (.dat, .hea, .atr)
├── Notebooks/                # Preprocessing, training & inference notebooks
├── Models/                   # Saved model files (.h5 / .keras / .pb)
├── Plots/                    # Accuracy, loss, and confusion matrix visualizations
├── data/                     # Processed feature and label arrays
├── requirements.txt           # Reproducible Python dependencies
├── LICENSE                    # MIT License
└── README.md
```

<div align="right">
  <a href="#top"><kbd> <br> 🡅 Back to Top <br> </kbd></a>
</div>

---

<!-- License -->
<a id="license"></a>
[![LICENSE](https://readme-typing-svg.herokuapp.com?font=JetBrainsMono+Nerd+Font&letterSpacing=0.3rem&pause=1000&width=400&lines=LICENSE)](https://git.io/typing-svg)

MIT License

Copyright (c) 2025 Neelotpal Santra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

<div align="right">
  <a href="#top"><kbd> <br> 🡅 Back to Top <br> </kbd></a>
</div>