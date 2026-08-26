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

The original CIFAR-10 training set was split into stratified
training, validation, and test sets:

- Train: 40,000 images
- Validation: 10,000 images
- Test: 10,000 images

The processed dataset is available as a GitHub Release:

[Download Dataset](https://github.com/VRYeshwanth/CIFAR10_Classifier/releases/tag/v1.0.0)