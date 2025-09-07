import os
import sys
import joblib
import logging
from sklearn.metrics import classification_report, accuracy_score
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_file import load_and_preprocess_data
from src.utils import read_config
from src.train import get_parsbert_embeddings

def evaluate_models():
    """Loads the test data and evaluates both trained models, printing a comparison report."""
    print("Starting model evaluation...")

    # Load configuration
    config = read_config()
    model_path = config.get('MODEL', 'model_path')
    _, X_test, _, y_test = load_and_preprocess_data(config)

    #  Evaluate TF-IDF + SVM Model 
    print(" Evaluating TF-IDF + SVM Model ")
    try:
        vectorizer = joblib.load(os.path.join(model_path, "tfidf_vectorizer.pkl"))
        tfidf_model = joblib.load(os.path.join(model_path, "tfidf_svm_model.pkl"))
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred_tfidf = tfidf_model.predict(X_test_tfidf)
        accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
        report_tfidf = classification_report(y_test, y_pred_tfidf, zero_division=0)
        
        print("\nTF-IDF + SVM Model Performance:")
        print(f"Accuracy: {accuracy_tfidf:.4f}")
        print("Classification Report:")
        print(report_tfidf)
    except FileNotFoundError:
        logging.error("TF-IDF model or vectorizer not found. Please train the model first.")

    #  Evaluate ParsBERT + SVM Model 
    print(" Evaluating ParsBERT + SVM Model ")
    try:
        parsbert_model = joblib.load(os.path.join(model_path, "parsbert_svm_model.pkl"))
        X_test_bert = get_parsbert_embeddings(X_test)
        y_pred_bert = parsbert_model.predict(X_test_bert)
        accuracy_bert = accuracy_score(y_test, y_pred_bert)
        report_bert = classification_report(y_test, y_pred_bert, zero_division=0)
        
        print("\nParsBERT + SVM Model Performance:")
        print(f"Accuracy: {accuracy_bert:.4f}")
        print("Classification Report:")
        print(report_bert)
    except FileNotFoundError:
        logging.error("ParsBERT SVM model not found. Please train the model first.")
        
    print("Model evaluation finished.")

if __name__ == '__main__':
    evaluate_models()
