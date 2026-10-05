# ECG Arrhythmia Classification using 1D-CNN

A deep learning project for classifying ECG heartbeats into five arrhythmia categories using a **1D Convolutional Neural Network (1D-CNN)** and the **MIT-BIH Arrhythmia Database**.

## 📌 Project Overview

Electrocardiograms (ECGs) contain valuable information about heart rhythm and cardiac activity. The goal of this project is to automatically classify individual ECG heartbeats into five classes using a 1D-CNN.

The project covers the complete machine learning pipeline:

**ECG recordings → preprocessing → heartbeat extraction → class mapping → CNN training → hyperparameter tuning → evaluation → explainability**

## 🎯 Classification Classes

The original MIT-BIH beat annotations were mapped into five AAMI-style categories:

| Class | Description |
|---|---|
| **N** | Normal |
| **S** | Supraventricular ectopic |
| **V** | Ventricular ectopic |
| **F** | Fusion |
| **Q** | Unknown / Paced |

## 📊 Dataset

The project uses the **MIT-BIH Arrhythmia Database**.

- 48 ECG records were downloaded using WFDB
- Paced records were excluded
- The **MLII lead** was selected when available
- ECG signals were filtered before heartbeat extraction
- **100,893 heartbeat samples** were obtained after preprocessing

### Class Distribution

| Class | Samples |
|---|---:|
| N | 90,094 |
| V | 7,008 |
| S | 2,781 |
| F | 802 |
| Q | 208 |
| **Total** | **100,893** |

The dataset is highly imbalanced, with Normal beats making up the majority of the samples. Class weighting and Macro-F1 were therefore used during model development and evaluation.

## ⚙️ Data Preprocessing

### 1. ECG Lead Selection

The **MLII lead** was selected from the ECG recordings so that the model received a consistent ECG signal type.

### 2. Band-pass Filtering

A **4th-order Butterworth band-pass filter** with a frequency range of **0.5–40 Hz** was applied to reduce baseline drift and high-frequency noise while retaining relevant ECG morphology.

### 3. R-Peak Based Segmentation

Heartbeat annotations were used to locate R-peaks.

For each heartbeat:

- 200 ms of signal before the R-peak was extracted
- 400 ms after the R-peak was extracted
- Total window = **600 ms**
- Each heartbeat was represented using **216 samples**

This produced a fixed-size input suitable for the 1D-CNN.

### 4. AAMI Class Mapping

Multiple MIT-BIH annotation symbols were grouped into the five target classes.

For example:

- `N, L, R, e, j → N`
- `A, a, J, S → S`
- `V, E → V`
- `F → F`
- `/, f, Q, x, X → Q`

### 5. Train/Validation/Test Split

The dataset was divided using a stratified:

- **80% training**
- **10% validation**
- **10% testing**

split with `random_state=42`.

## 🧠 Model Architecture

A **1D Convolutional Neural Network** was used because ECG heartbeat classification depends strongly on local waveform morphology.

The final tuned architecture contains:

```text
Input ECG (216 samples)
        ↓
Conv1D – 16 filters
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv1D – 32 filters
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv1D – 64 filters
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv1D – 128 filters
        ↓
Batch Normalization
        ↓
Global Average Pooling
        ↓
Dense – 64
        ↓
Dropout – 0.3
        ↓
Softmax – 5 classes
```

### Why 1D-CNN?

CNNs can learn local patterns directly from the ECG waveform.

Earlier convolutional layers can learn simpler waveform patterns, while deeper layers can combine these into more complex morphological features.

A 1D-CNN is also suitable for fixed-length ECG heartbeat segments because the model can operate directly along the time dimension without requiring manual feature engineering.

## 🔧 Training

The model was trained using:

- **TensorFlow/Keras**
- **Adam optimizer**
- **Categorical cross-entropy loss**
- Batch size: **64**
- Maximum epochs: **30**
- Early stopping with patience of **5 epochs**
- Best weights restored after early stopping
- **Class weights** to address class imbalance

## 🔍 Hyperparameter Tuning

**Keras Tuner Random Search** was used to explore different CNN configurations.

The search included:

- Number of CNN blocks: **2–4**
- Base filters: **16, 32, 64**
- Kernel size: **3, 5, 7, 9, 11**
- Dense units: **64, 128, 256**
- Dropout: **0.2–0.5**
- Learning rate: approximately **1e-4 to 1e-3**

A total of **15 trials** were performed.

The tuning objective was **validation Macro-F1**, rather than validation accuracy, because of the severe class imbalance.

### Best Configuration

| Hyperparameter | Value |
|---|---:|
| CNN blocks | 4 |
| Base filters | 16 |
| Kernel size | 11 |
| Dense units | 64 |
| Dropout | 0.3 |
| Learning rate | ~0.0001676 |
| Validation Macro-F1 | ~0.895 |

## 📈 Results

The final model achieved the following performance on the held-out test set:

| Metric | Result |
|---|---:|
| **Test Accuracy** | **97.78%** |
| **Macro-F1** | **84.54%** |
| **Macro ROC-AUC** | **~0.9971** |

### Class-wise F1 Scores

| Class | F1 Score |
|---|---:|
| F | 66.09% |
| N | 98.90% |
| Q | 78.26% |
| S | 83.92% |
| V | 95.52% |

The difference between accuracy and Macro-F1 highlights the effect of class imbalance: the model performs extremely well overall, particularly on the majority Normal class, while the minority classes remain more challenging.

## 🧪 Filtering Ablation Study

To evaluate whether ECG filtering contributed to model performance, the model was also evaluated without Butterworth filtering.

| Setup | Accuracy | Macro-F1 |
|---|---:|---:|
| **Filtered ECG** | **97.78%** | **84.54%** |
| Raw ECG | 96.94% | 80.50% |

Removing filtering resulted in:

- **0.84 percentage-point decrease in accuracy**
- **4.04 percentage-point decrease in Macro-F1**

This suggests that preprocessing had a meaningful effect, particularly on balanced performance across classes.

## 🔬 Explainability

Two explainability approaches were explored.

### Grad-CAM

**Grad-CAM** was used to identify important temporal regions of an ECG waveform that contributed to a CNN prediction.

This provides a heatmap over the ECG signal and helps visualize which parts of the heartbeat the convolutional network considered important.

### SHAP

**SHAP** was used to estimate the contribution of individual ECG input points toward model predictions.

Together, Grad-CAM and SHAP provide complementary views of model behavior:

- **Grad-CAM:** highlights important regions of the ECG waveform
- **SHAP:** estimates contribution of individual input features/sample points

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **SciPy**
- **WFDB**
- **NumPy**
- **Scikit-learn**
- **Keras Tuner**
- **SHAP**
- **Matplotlib**

## 📁 Project Workflow

```text
MIT-BIH Arrhythmia Database
            ↓
       Load ECG Records
            ↓
      Select MLII Lead
            ↓
   Butterworth Filtering
            ↓
       Find R-Peaks
            ↓
    Extract 600 ms Beats
            ↓
      AAMI Class Mapping
            ↓
     Stratified Data Split
            ↓
       Class Weighting
            ↓
     1D-CNN Training
            ↓
     Hyperparameter Tuning
            ↓
       Model Evaluation
            ↓
   Grad-CAM + SHAP Analysis
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install dependencies

```bash
pip install numpy scipy scikit-learn tensorflow keras-tuner wfdb shap matplotlib
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open:

```text
model_training(1).ipynb
```

### 4. Run the notebook

Run the cells sequentially to:

1. Download the MIT-BIH data
2. Preprocess ECG signals
3. Extract heartbeat segments
4. Create training/validation/test datasets
5. Train the CNN
6. Perform hyperparameter tuning
7. Evaluate the model
8. Run explainability analysis

## ⚠️ Limitations

The reported results should be interpreted as **experimental machine-learning results**, not clinical validation.

One important limitation is that the dataset was split at the **heartbeat level** rather than strictly at the patient/record level. Because multiple heartbeats can originate from the same patient, this setup may produce correlated samples across training and test sets.

A stronger evaluation would use:

- Patient/record-level splitting
- External validation on an independent ECG dataset
- Additional clinical evaluation

Therefore, the reported **97.78% accuracy should not be interpreted as evidence of clinical readiness**.

## 📌 Key Takeaways

- Built an end-to-end ECG classification pipeline using a **1D-CNN**
- Processed **100K+ heartbeat segments**
- Addressed severe class imbalance using **class weights and Macro-F1**
- Used **Keras Tuner** for architecture and hyperparameter optimization
- Achieved **97.78% test accuracy and 84.54% Macro-F1**
- Demonstrated the effect of ECG filtering through an **ablation study**
- Used **Grad-CAM and SHAP** for model explainability
