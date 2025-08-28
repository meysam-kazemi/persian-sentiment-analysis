# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os

# Import the preprocessing function from the other file
from preprocess import clean_text

# Path to the data file
DATA_PATH = 'data/sample_comments.csv'
MODELS_DIR = 'models'

def train_model():
    """
    The main function to train and evaluate the model.
    """
    # 1. Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Check for required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must contain 'text' and 'label' columns.")

    # 2. Preprocess text data
    print("Preprocessing text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # 3. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label'] # To maintain the class ratio
    )

    # 4. Feature extraction using TF-IDF
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Train the Naive Bayes model
    print("Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 6. Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save the model and vectorizer
    print("Saving model and vectorizer...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, 'sentiment_model.joblib'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer.joblib'))
    
    print("\nTraining complete! Model saved in 'models/' directory.")

if __name__ == '__main__':
    train_model()
