import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re
import nltk
import joblib  # <-- New import
import json    # <-- New import
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- NLTK Downloader ---
# This is still needed for Streamlit Cloud to preprocess user input
try:
    stopwords.words('english')
except LookupError:
    st.write("Downloading NLTK 'stopwords' resource...")
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.write("Downloading NLTK 'punkt' resource...")
    nltk.download('punkt')
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    st.write("Downloading NLTK 'punkt_tab' resource...")
    nltk.download('punkt_tab')

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🤖",
    layout="wide"
)

# --- Caching Functions for Performance ---

@st.cache_data
def load_data(filepath):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath, dtype={'review': 'str', 'sentiment': 'str'})
        df.columns = df.columns.str.lower()
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please ensure it's in the same directory as the app.")
        return None

# Load the saved model and vectorizer
@st.cache_resource
def load_model_assets():
    """Loads the saved model and vectorizer from disk."""
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return model, vectorizer

# Load the saved evaluation metrics
@st.cache_data
def load_evaluation_metrics():
    """Loads the saved evaluation metrics from a JSON file."""
    with open('evaluation_metrics.json', 'r') as f:
        metrics = json.load(f)
    return metrics

# --- Text Preprocessing Function ---
# This is still needed to process *new* user input
def preprocess_text(text):
    """Cleans and prepares text data for modeling."""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


# --- Load all assets at startup ---
model, vectorizer = load_model_assets()
metrics = load_evaluation_metrics()
df = load_data('IMDB Dataset.csv') # Still needed for visualization

# --- Main Application ---
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("This application analyzes text to determine if its sentiment is **positive** or **negative** using Natural Language Processing (NLP).")

# --- 1. Project Documentation ---
with st.expander("Show Project Documentation", expanded=False):
    st.header("Project Documentation")
    # (Your documentation content remains unchanged)
    st.subheader("a) Problem Statement")
    st.write("""
    The goal of this project is to build and evaluate a machine learning model capable of performing sentiment analysis.
    Specifically, the model will classify a given text (in this case, a movie review) as having either a positive or
    negative sentiment. This demonstrates a foundational skill in data analytics: extracting subjective meaning from unstructured text data.
    """)
    st.subheader("b) Algorithm of the Solution")
    st.write("""
    The solution follows these key steps:
    1.  **Data Loading & Exploration**: The "IMDB Dataset of 50k Movie Reviews" is loaded to visualize the sentiment distribution.
    2.  **Model Training (Offline)**: A separate script (`train_model.py`) is used to preprocess the text, split the data 80:20, and train a TF-IDF Vectorizer and a Logistic Regression model.
    3.  **Model Persistence**: The trained vectorizer and model, along with their evaluation metrics, are saved to disk (`.joblib` and `.json` files).
    4.  **Model Loading**: This Streamlit app loads the pre-trained files at startup.
    5.  **Evaluation Display**: The app displays the pre-computed accuracy, confusion matrix, and classification report.
    """)
    st.subheader("c) Analysis of Findings")
    st.write("""
    The analysis of the model's performance is detailed in the 'Model Evaluation' section below. It includes the overall accuracy score, a confusion matrix to visualize correct and incorrect predictions, and a classification report with precision, recall, and F1-scores. This comprehensive evaluation determines the model's effectiveness.
    """)
    st.subheader("d) References")
    st.write("""
    - **Dataset**: [IMDB Dataset of 50k Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
    - **Libraries**: Pandas, NLTK, Scikit-learn, Streamlit, Matplotlib, Seaborn, BeautifulSoup, Joblib.
    """)

# --- 1. Data Exploration and Visualization ---
st.header("1. Data Exploration and Visualization")
if df is not None:
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    st.write("Descriptive Statistics:")
    st.write(df.describe(include='all'))

    # --- Data Visualization ---
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='sentiment', hue='sentiment', ax=ax, palette="viridis", legend=False)    
    ax.set_title("Distribution of Positive and Negative Reviews")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.write("The dataset is perfectly balanced, with an equal number of positive and negative reviews.")

# --- 2. Build and Evaluate the Model ---
st.header("2. Build and Evaluate the Model")
st.write("The model was trained offline. The evaluation metrics from the test set are loaded from disk and displayed below.")

# Display metrics loaded from our JSON file
st.subheader("Model Performance on Test Data")
st.metric(label="Model Accuracy", value=f"{metrics['accuracy']:.2%}")

# Display Confusion Matrix
st.subheader("Confusion Matrix")
st.write("The confusion matrix shows the number of correct and incorrect predictions made by the model.")
cm = np.array(metrics['confusion_matrix']) # Convert list back to numpy array
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# Display Performance Metrics
st.subheader("Classification Report")
# Re-create the text report from the saved dictionary
report_text = f"""
              precision    recall  f1-score   support

    Negative       {metrics['classification_report']['Negative']['precision']:.2f}         {metrics['classification_report']['Negative']['recall']:.2f}      {metrics['classification_report']['Negative']['f1-score']:.2f}     {metrics['classification_report']['Negative']['support']}
    Positive       {metrics['classification_report']['Positive']['precision']:.2f}         {metrics['classification_report']['Positive']['recall']:.2f}      {metrics['classification_report']['Positive']['f1-score']:.2f}     {metrics['classification_report']['Positive']['support']}

    accuracy                           {metrics['accuracy']:.2f}     {metrics['classification_report']['macro avg']['support']}
   macro avg       {metrics['classification_report']['macro avg']['precision']:.2f}         {metrics['classification_report']['macro avg']['recall']:.2f}      {metrics['classification_report']['macro avg']['f1-score']:.2f}     {metrics['classification_report']['macro avg']['support']}
weighted avg       {metrics['classification_report']['weighted avg']['precision']:.2f}         {metrics['classification_report']['weighted avg']['recall']:.2f}      {metrics['classification_report']['weighted avg']['f1-score']:.2f}     {metrics['classification_report']['weighted avg']['support']}
"""
st.text(report_text)

# --- 3. Make Predictions ---
st.header("3. Make Your Own Prediction")
st.write("Enter some text below to see how the model classifies its sentiment.")
user_input = st.text_area("Enter text for sentiment analysis:", "This movie was absolutely fantastic! The acting was superb and the plot was gripping.")

if st.button("Analyze Sentiment"):
    if user_input:
        # 1. Preprocess the user's input
        cleaned_input = preprocess_text(user_input)
        
        # 2. Vectorize the input using the FITTED vectorizer
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # 3. Predict using the TRAINED model
        prediction = model.predict(vectorized_input)
        prediction_proba = model.predict_proba(vectorized_input)

        # 4. Display the result
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        confidence = prediction_proba[0][prediction[0]]

        if sentiment == "Positive":
            st.success(f"**Predicted Sentiment: {sentiment}** (Confidence: {confidence:.2%}) 👍")
        else:
            st.error(f"**Predicted Sentiment: {sentiment}** (Confidence: {confidence:.2%}) 👎")
    else:
        st.warning("Please enter some text to analyze.")

# --- 4. Project Summary ---
st.header("4. Project Summary and Conclusion")
# (Your summary content remains unchanged)
st.write("""
This project successfully demonstrates the creation of a sentiment analysis model using Python and the NLTK/Scikit-learn libraries. The Logistic Regression model, trained on the TF-IDF vectorized IMDB movie review dataset, achieved a high accuracy, indicating its effectiveness in distinguishing between positive and negative sentiment.

**Suitability:**
- The model is **highly suitable** for applications where a binary sentiment (positive/negative) is sufficient, such as product review analysis, social media monitoring, and customer feedback classification.
- The TF-IDF approach is effective because it captures word importance, making the model robust.

**Limitations:**
- The model is **less suitable** for detecting more nuanced sentiments like neutrality, sarcasm, or irony, as it was only trained on positive and negative examples.
- Its performance is dependent on the vocabulary present in the training data. It may struggle with new slang, domain-specific jargon, or evolving language.
""")