import os
import sys
import numpy as np
import torch
import joblib
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_file import load_and_preprocess_data
from src.utils import read_config
from src.train_models import get_parsbert_embeddings
from src.bert_classifier import BertClassifier

def evaluate_models():
    """Loads the test data and evaluates both trained models, printing a comparison report."""
    print("Starting model evaluation...")

    # Load configuration
    config = read_config()
    model_path = config.get('MODEL', 'model_path')
    _, X_test, _, y_test = load_and_preprocess_data(config)

    #  Evaluate TF-IDF + SVM Model 
    print(" Evaluating TF-IDF + SVM Model ")
    vectorizer = joblib.load(os.path.join(model_path, "tfidf_vectorizer.pkl"))
    tfidf_model = joblib.load(os.path.join(model_path, "tfidf_svm_model.pkl"))
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred_tfidf = tfidf_model.predict(X_test_tfidf)
    accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
    auc_tfidf = roc_auc_score(y_test, y_pred_tfidf)
    f1_tfidf = f1_score(y_test, y_pred_tfidf)
    report_tfidf = classification_report(y_test, y_pred_tfidf, zero_division=0)
    
    print("\nTF-IDF + SVM Model Performance:")
    print(f"Accuracy: {accuracy_tfidf:.4f}")
    print(f"AUC: {auc_tfidf:.4f}")
    print(f"F1: {f1_tfidf:.4f}")
    print("Classification Report:")
    print(report_tfidf)

    print(" Evaluating ParsBERT + NN Model ")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifier()
    model_save_path = os.path.join(model_path, "bert_nn_model.pth")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    X_test_bert = get_parsbert_embeddings(X_test)

    X_test_tensor = torch.tensor(X_test_bert, dtype=torch.float32).to(device)
    y_pred_bert = []
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(X_test_tensor)
        # Get the predicted class index with the highest score
        _, predicted_indices = torch.max(outputs, dim=1)
        y_pred_bert.extend(predicted_indices.cpu().numpy())
    y_pred_bert = np.array(y_pred_bert)

    accuracy_bert = accuracy_score(y_test, y_pred_bert)
    auc_bert = roc_auc_score(y_test, y_pred_bert)
    f1_bert = f1_score(y_test, y_pred_bert)
    report_bert = classification_report(y_test, y_pred_bert, zero_division=0)
    
    print("\nParsBERT + NN Model Performance:")
    print(f"Accuracy: {accuracy_bert:.4f}")
    print(f"AUC: {auc_bert:.4f}")
    print(f"F1: {f1_bert:.4f}")
    print("Classification Report:")
    print(report_bert)
        
    print("Model evaluation finished.")

if __name__ == '__main__':
    evaluate_models()
