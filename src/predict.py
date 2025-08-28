# src/predict.py

import joblib
import argparse
import os
from preprocess import clean_text

# Path to the models directory
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_model.joblib')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.joblib')

def predict_sentiment(text):
    """
    Predicts the sentiment of an input text.
    """
    # Check if model files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return "Model files not found. Please run 'train.py' first."

    # 1. Load the model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # 2. Preprocess the input text
    cleaned_text = clean_text(text)
    
    # 3. Transform the text into a numerical vector
    text_vec = vectorizer.transform([cleaned_text])
    
    # 4. Make a prediction
    prediction = model.predict(text_vec)
    
    return prediction[0]

if __name__ == '__main__':
    # Use argparse to get text from the command line
    parser = argparse.ArgumentParser(description="Predict sentiment of a given Persian text.")
    parser.add_argument('--text', type=str, required=True, help='The text to analyze.')
    
    args = parser.parse_args()
    
    # Call the prediction function and print the result
    sentiment = predict_sentiment(args.text)
    print(f"Predicted sentiment: {sentiment}")
