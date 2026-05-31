# Reliability-Aware Skin Cancer Classification

A trust-aware deep learning system for **skin cancer classification** using **ResNet18** and **Monte Carlo Dropout–based uncertainty estimation** on the HAM10000 dermoscopic image dataset.

The project goes beyond standard classification by introducing a **Reliability Index**, enabling confidence-aware predictions for safer AI-assisted medical decision-making.

---

## Project Overview

Traditional medical AI systems often provide predictions without indicating **how confident or reliable** they are. In healthcare, unreliable predictions can be dangerous.

This project addresses that challenge by combining:

- **Transfer Learning (ResNet18)** for skin lesion classification  
- **Monte Carlo Dropout** for uncertainty estimation  
- **Reliability Index** for trust-aware prediction filtering  
- **Selective Prediction Framework** to improve diagnostic confidence

The system predicts whether a lesion is:

- **Benign**
- **Malignant**

while also estimating **prediction confidence and reliability**.

---

## Features

- Deep learning–based skin cancer classification using **PyTorch**
- **Transfer learning** with pretrained ResNet18
- **Monte Carlo Dropout (MC Dropout)** for uncertainty estimation
- Reliability-aware inference pipeline
- Confidence-based selective prediction
- Performance visualization and evaluation metrics
- Threshold-based reliability analysis

---

## Tech Stack

**Languages & Frameworks**
- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn

**Deep Learning**
- ResNet18
- Transfer Learning
- Weighted Cross Entropy Loss

**Dataset**
- HAM10000 (Human Against Machine with 10,000 training images)

---

## Project Structure

```txt
Reliability-Aware-Skin-Cancer-Classification/
│
├── data/                     # Dataset placeholder
│
├── models/
│   └── README.md             # Model loading instructions
│
├── notebooks/                # Experiment notebooks
│
├── results/
│   ├── figure1_threshold_vs_metrics.png
│   ├── figure2_reliability_tradeoff.png
│   ├── figure3_confusion_matrix.png
│   ├── figure4_reliability_histogram.png
│   ├── final_summary.csv
│   └── multi_run_results.csv
│
├── src/
│   ├── preprocess_dataset.py
│   ├── data_loader.py
│   ├── train_resnet.py
│   ├── mc_dropout.py
│   ├── evaluation.py
│   ├── plots.py
│   └── results.py
│
└── README.md
```

---

## Methodology

### 1. Data Preprocessing
- Processed the **HAM10000 dermoscopic dataset**
- Created **train / validation / test splits**
- Performed image transformations and normalization

### 2. Model Training
- Used **ResNet18 pretrained on ImageNet**
- Replaced final classification layer for **binary classification**
- Applied **weighted cross-entropy loss** to handle imbalance

### 3. Reliability Estimation
Instead of relying on a single prediction, the model performs **multiple stochastic forward passes** using **Monte Carlo Dropout**.

This allows the system to compute:

- **Prediction Confidence**
- **Uncertainty (Variance)**
- **Reliability Score**

### 4. Selective Prediction
Low-confidence predictions can be filtered, improving trustworthiness for clinical decision support.

---

## Results

### Model Performance
- **Validation Accuracy:** 89%
- **Confidence-aware Prediction System**
- **Monte Carlo Dropout–based Reliability Estimation**

### Reliability Improvements
- Improved filtered diagnostic accuracy from **67.7% → 74.8%**
- Achieved **80%+ accuracy under high-confidence thresholds**

---

## Example Output

```python
===== FINAL RESULT =====
Prediction: Benign
Confidence: 0.9979
Reliability: 1.0
```

---

## Visualizations

The project includes:

- Threshold vs Metrics Analysis
- Reliability–Accuracy Tradeoff
- Confusion Matrix
- Reliability Distribution Histogram

All plots are available in the `results/` directory.

---

## How to Run

### 1. Clone the Repository

```bash
git clone <your_repo_link>
cd Reliability-Aware-Skin-Cancer-Classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python src/train_resnet.py
```

### 4. Run Reliability Inference

```bash
python src/mc_dropout.py
```

### 5. Evaluate Results

```bash
python src/evaluation.py
```

---

## Future Improvements

- Vision Transformer (ViT)-based architecture
- Multi-class skin lesion classification
- Explainable AI (Grad-CAM)
- Clinical deployment interface
- Real-time diagnostic dashboard

---

## Author

**Tanisha Kumar**  
B.Tech CSE (AI & ML)
