# 🧠 CIFAR-10 Classifier

A modular PyTorch project for experimenting with Convolutional Neural Networks (CNNs) on the CIFAR-10 image classification dataset.

## 📂 Project Structure

```
CIFAR10_Classifier/
├── images/
│   └── baseline/
│       ├── Model Performance.png
│       └── Training and Validation Metrics.png
├── models/
│   └── CIFAR_Baseline.pt
├── notebooks/
│   ├── cnn.ipynb
│   └── data_exploration.ipynb
├── README.md
├── requirements.txt
└── src/
    ├── dataset.py
    ├── engine.py
    ├── metrics.py
    ├── models.py
    └── plots.py
```

## 🛠️ Tech Stack

- Python
- PyTorch
- TorchVision
- TorchMetrics
- Pandas
- Matplotlib
- Jupyter Notebook

## ⚙️ Setup

Clone the repository:

```bash
git clone https://github.com/VRYeshwanth/CIFAR10_Classifier.git
cd CIFAR10_Classifier
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / MacOS**
```bash
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

Download the CIFAR-10 dataset and place the extracted training images and labels inside:
```
cifar-10/
├── train/
└── trainLabels.csv
```

## 📦 Dataset

This project uses the CIFAR-10 training dataset. The original training set is split into three subsets:

- Train: 40,000 images
- Validation: 10,000 images
- Test: 10,000 images

The original CIFAR-10 test set is not used in this project.

> Note: Since the test set is derived from the original training data, these results should not be directly compared with benchmarks that use the official CIFAR-10 test set.