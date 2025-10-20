import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- NLTK Downloader ---
# We check for and download all required NLTK data packages.
# This runs every time the app starts on Streamlit Cloud.
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
    # This is the one that was missing!
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
# Cache data loading
@st.cache_data
def load_data(filepath):
    """Loads the dataset from a CSV file."""
    try:
        # THIS IS THE FIX: Explicitly set dtypes to stop pyarrow errors
        df = pd.read_csv(filepath, dtype={'review': 'str', 'sentiment': 'str'})

        df.columns = df.columns.str.lower()
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please ensure it's in the same directory as the app.")
        return None

# Cache model training
@st.cache_resource
def train_model(X, y):
    """Preprocesses data, trains the TF-IDF vectorizer and Logistic Regression model."""
    # Split data into training and testing sets (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Initialize and train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer, X_test_tfidf, y_test

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """Cleans and prepares text data for modeling."""
    # 1. Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

    # 3. Convert to lowercase
    text = text.lower()

    # 4. Tokenize text
    tokens = word_tokenize(text)

    # 5. Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # 6. Join tokens back into a string
    return " ".join(filtered_tokens)


# --- Main Application ---
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("This application analyzes text to determine if its sentiment is **positive** or **negative** using Natural Language Processing (NLP).")

# --- 1. Project Documentation ---
with st.expander("Show Project Documentation", expanded=False):
    st.header("Project Documentation")

    st.subheader("a) Problem Statement")
    st.write("""
    The goal of this project is to build and evaluate a machine learning model capable of performing sentiment analysis.
    Specifically, the model will classify a given text (in this case, a movie review) as having either a positive or
    negative sentiment. This demonstrates a foundational skill in data analytics: extracting subjective meaning from unstructured text data.
    """)

    st.subheader("b) Algorithm of the Solution")
    st.write("""
    The solution follows these key steps:
    1.  **Data Loading & Exploration**: The "IMDB Dataset of 50k Movie Reviews" is loaded. A descriptive analysis is performed to understand its structure and check for missing values.
    2.  **Data Preprocessing**: Each review undergoes a cleaning process to:
        - Remove HTML tags.
        - Eliminate punctuation and special characters.
        - Convert all text to lowercase.
        - Remove common English "stop words" (e.g., 'the', 'a', 'is') that don't carry significant sentiment.
    3.  **Feature Extraction (Vectorization)**: The cleaned text is converted into a numerical format using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique. This method scores words based on their importance in a document relative to the entire corpus.
    4.  **Model Training**: The dataset is split into 80% for training and 20% for testing. A **Logistic Regression** algorithm, a binary classification model, is trained on the vectorized training data.
    5.  **Model Evaluation**: The model's performance is assessed on the unseen test data using metrics like accuracy and a confusion matrix.
    """)

    st.subheader("c) Analysis of Findings")
    st.write("""
    The analysis of the model's performance is detailed in the 'Model Evaluation' section below. It includes the overall accuracy score, a confusion matrix to visualize correct and incorrect predictions, and a classification report with precision, recall, and F1-scores. This comprehensive evaluation determines the model's effectiveness.
    """)

    st.subheader("d) References")
    st.write("""
    - **Dataset**: [IMDB Dataset of 50k Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
    - **Libraries**: Pandas, NLTK, Scikit-learn, Streamlit, Matplotlib, Seaborn, BeautifulSoup.
    """)

# --- Load and Display Data ---
st.header("1. Data Exploration and Visualization")
df = load_data('IMDB Dataset.csv')

if df is not None:
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    st.write("Descriptive Statistics:")
    st.write(df.describe(include='all'))

    # --- Data Visualization ---
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    # NEW LINE
    sns.countplot(data=df, x='sentiment', hue='sentiment', ax=ax, palette="viridis", legend=False)    
    ax.set_title("Distribution of Positive and Negative Reviews")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.write("The dataset is perfectly balanced, with an equal number of positive and negative reviews.")

    # --- 2. Model Building ---
    st.header("2. Build and Evaluate the Model")
    st.write("The model is built by preprocessing the text, vectorizing it with TF-IDF, and training a Logistic Regression classifier. This may take a moment...")

    # Preprocess data
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    # Map sentiment to numerical values for the model
    df['sentiment_numeric'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X = df['cleaned_review']
    y = df['sentiment_numeric']

    # Train the model and get results
    model, vectorizer, X_test_tfidf, y_test = train_model(X, y)

    # --- 3. Model Evaluation ---
    st.subheader("Model Performance on Test Data")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")

    # Display Confusion Matrix
    st.subheader("Confusion Matrix")
    st.write("The confusion matrix shows the number of correct and incorrect predictions made by the model.")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # Display Performance Metrics
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    st.text(report)


    # --- 4. Make Predictions ---
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

# --- 5. Project Summary ---
st.header("4. Project Summary and Conclusion")
st.write("""
This project successfully demonstrates the creation of a sentiment analysis model using Python and the NLTK/Scikit-learn libraries. The Logistic Regression model, trained on the TF-IDF vectorized IMDB movie review dataset, achieved a high accuracy, indicating its effectiveness in distinguishing between positive and negative sentiment.

**Suitability:**
- The model is **highly suitable** for applications where a binary sentiment (positive/negative) is sufficient, such as product review analysis, social media monitoring, and customer feedback classification.
- The TF-IDF approach is effective because it captures word importance, making the model robust.

**Limitations:**
- The model is **less suitable** for detecting more nuanced sentiments like neutrality, sarcasm, or irony, as it was only trained on positive and negative examples.
- Its performance is dependent on the vocabulary present in the training data. It may struggle with new slang, domain-specific jargon, or evolving language.
""")