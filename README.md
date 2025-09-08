# Snappfood Sentiment Analysis 🍲

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: apache](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/meysam-kazemi/image-classification/blob/main/LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased)

A machine learning project to classify the sentiment of user comments from Snappfood. This repository provides a full pipeline from data preprocessing to model training, evaluation, and an interactive web demo. It compares a classic machine learning model (TF-IDF + SVM) with a modern transformer-based approach (ParsBERT + NN).

![vid](https://github.com/meysam-kazemi/persian-sentiment-analysis/blob/main/vid/sentim_.gif)

![vid](https://github.com/meysam-kazemi/persian-sentiment-analysis/blob/main/vid/sentim.webm)
---

## ✨ Features

- **Persian Text Analysis**: Specifically tailored for analyzing Persian-language comments.
- **Dual-Model Comparison**: Implements and evaluates two distinct NLP approaches for a comprehensive comparison.
- **Modular Codebase**: The project is structured into logical modules for data processing, training, and evaluation.
- **Configuration Driven**: All key parameters and paths are managed via a central `config.ini` file.
- **Interactive Demo**: Includes a web application built with Gradio to test the model live.

---

## 📂 Project Structure

The repository is organized to ensure clarity and scalability.

```
├── app.py
├── config.ini
├── data
│   ├── snappfood-processed
│   └── Snappfood - Sentiment Analysis.csv
├── LICENSE
├── models
│   ├── bert_nn_model.pth
│   ├── tfidf_svm_model.pkl
│   └── tfidf_vectorizer.pkl
├── README.md
├── requirements.txt
├── src
│   ├── bert_classifier.py
│   ├── evaluator.py
│   ├── read_file.py
│   ├── train_models.py
│   └── utils.py
└── vid
    └── sentim_.gif
```

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:meysam-kazemi/persian-sentiment-analysis.git
    cd persian-sentiment-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Before running the scripts, review the `config.ini` file. Ensure the paths and column names match your dataset.

```ini
[DATA]
path=data/Snappfood - Sentiment Analysis.csv
processed_path=data/processed
model_path=models/
...

[DF]
text_col=comment
label_col=label_id
...
```

---

## ⚙️ Usage

The project pipeline is executed through the scripts in the `src/` directory.

1.  **Process the Data:**
    This script reads the raw CSV, cleans it, and splits it into training and testing sets.
    ```bash
    python src/read_file.py
    ```

2.  **Train the Models:**
    This script trains both the TF-IDF+SVM and ParsBERT+SVM models and saves them to the `models/` directory.
    *Note: Training the ParsBERT model is computationally intensive and may take a long time without a GPU.*
    ```bash
    python src/train_models.py
    ```

3.  **Evaluate the Models:**
    This script evaluates the performance of the trained models on the test set and prints a comparison report.
    ```bash
    python src/evaluator.py
    ```

4.  **Launch the Web App:**
    To test the TF-IDF model interactively, run the Gradio app.
    ```bash
    gradio app.py
    ```
    Open the URL provided in your terminal (e.g., `http://127.0.0.1:7860`) in a browser.

---

## 📄 License

This project is licensed under the Apache License. See the `LICENSE` file for more details.
