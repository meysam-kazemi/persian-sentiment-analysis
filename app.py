import gradio as gr
import joblib
import os
from src.utils import read_config

#  1. Load Configuration and Models 
try:
    # Read configuration to find model paths
    config = read_config()
    model_path = config.get('MODEL', 'model_path')

    # Construct full paths to the model files
    vectorizer_file = os.path.join(model_path, 'tfidf_vectorizer.pkl')
    model_file = os.path.join(model_path, 'tfidf_svm_model.pkl')

    # Load the TF-IDF vectorizer and the SVM model
    vectorizer = joblib.load(vectorizer_file)
    model = joblib.load(model_file)
    print("Models and vectorizer loaded successfully!")

except FileNotFoundError:
    print("Error: Model or vectorizer files not found.")
    print("Please run the training script first to generate model files.")
    vectorizer = None
    model = None

#  2. Define Prediction Function 
def predict_sentiment(comment):
    """
    Predicts the sentiment of a given comment using the loaded TF-IDF model.
    """
    if not model or not vectorizer:
        return "Model not loaded. Please check the console for errors."

    # Preprocess the input comment using the loaded vectorizer
    comment_vector = vectorizer.transform([comment])

    # Predict the sentiment
    prediction = model.predict(comment_vector)[0]

    if prediction == 1:
        sentiment = "Positive"
    elif prediction == 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral" # Or handle other cases as needed

    # Return a dictionary for the Gradio Label component
    return {sentiment: 1.0}


#  3. Create and Launch the Gradio Interface 
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="...این رستوران کیفیت غذای خوبی داشت",
        label="Enter a Snappfood Comment (in Persian)"
    ),
    outputs=gr.Label(num_top_classes=1, label="Predicted Sentiment"),
    title="Snappfood Sentiment Analysis 🍲",
    description="A simple app to predict the sentiment of Snappfood comments using a TF-IDF + SVM model. Type a comment and click 'Submit'.",
    examples=[
        ["غذا خیلی دیر رسید و کاملا سرد بود"],
        ["کیفیت غذا و بسته‌بندی عالی بود، حتما دوباره سفارش میدم"],
        ["قیمت‌ها کمی بالا بود ولی در کل خوب بود"]
    ]
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
