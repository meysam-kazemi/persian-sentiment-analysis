import os
import sys
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import joblib
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_file import load_and_preprocess_data
from src.utils import read_config

# Load configuration once
config = read_config()

def get_parsbert_embeddings(texts):
    """Generates sentence embeddings for a list of texts using ParsBERT."""
    print("Initializing ParsBERT model and tokenizer...")
    parsbert_model_name = config.get('MODEL', 'parsbert_model_name')
    tokenizer = BertTokenizer.from_pretrained(parsbert_model_name)
    model = BertModel.from_pretrained(parsbert_model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    model.eval()
    embeddings = []
    
    print("Generating ParsBERT embeddings...")
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Embeddings"):
        batch_texts = texts[i:i+batch_size].tolist() if isinstance(texts, pd.Series) else texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
        
    return np.vstack(embeddings)

def train_tfidf_svm(X_train, y_train):
    """Trains an SVM model using TF-IDF features and saves the model and vectorizer."""
    print("Starting TF-IDF + SVM model training...")
    
    model_path = config.get('MODEL', 'model_path')
    random_state = config.getint('MODEL', 'random_state')
    
    os.makedirs(model_path, exist_ok=True)

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    joblib.dump(vectorizer, os.path.join(model_path, "tfidf_vectorizer.pkl"))
    print(f"TF-IDF vectorizer saved to {model_path}")

    print("Training SVM model on TF-IDF features...")
    svm_model = SVC(kernel='linear', random_state=random_state)
    svm_model.fit(X_train_tfidf, y_train)
    joblib.dump(svm_model, os.path.join(model_path, "tfidf_svm_model.pkl"))
    print(f"TF-IDF + SVM model saved to {model_path}")
    print("TF-IDF + SVM training finished.")

def train_parsbert_svm(X_train, y_train):
    """Trains an SVM model using ParsBERT embeddings and saves the model."""
    print("Starting ParsBERT + SVM model training...")

    model_path = config.get('MODEL', 'model_path')
    random_state = config.getint('MODEL', 'random_state')
    
    os.makedirs(model_path, exist_ok=True)

    X_train_bert = get_parsbert_embeddings(X_train)
    
    print("Training SVM model on ParsBERT features...")
    svm_model = SVC(kernel='linear', random_state=random_state)
    svm_model.fit(X_train_bert, y_train)
    joblib.dump(svm_model, os.path.join(model_path, "parsbert_svm_model.pkl"))
    print(f"ParsBERT + SVM model saved to {model_path}")
    print("ParsBERT + SVM training finished.")

if __name__ == '__main__':
    X_train, _, y_train, _ = load_and_preprocess_data(config)
    train_tfidf_svm(X_train, y_train)
    train_parsbert_svm(X_train, y_train)
