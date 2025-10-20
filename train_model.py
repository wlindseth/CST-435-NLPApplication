import pandas as pd
import numpy as np
import re
import nltk
import json
import joblib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Ensure NLTK data is downloaded ---
# You already ran this locally, but it's good practice
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# --- Text Preprocessing Function ---
# (Copied from your app.py)
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

print("Starting model training process...")

# --- 1. Load Data ---
print("Loading dataset...")
df = pd.read_csv('IMDB Dataset.csv', dtype={'review': 'str', 'sentiment': 'str'})

# --- 2. Preprocess Data ---
print("Preprocessing text data...")
df['cleaned_review'] = df['review'].apply(preprocess_text)
df['sentiment_numeric'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Handle any potential NaN values that resulted from preprocessing
df.dropna(subset=['cleaned_review', 'sentiment_numeric'], inplace=True)

X = df['cleaned_review']
y = df['sentiment_numeric'].astype(int) # Ensure y is integer

# --- 3. Split Data ---
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Vectorize Text ---
print("Training TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- 5. Train Model ---
print("Training Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# --- 6. Evaluate Model ---
print("Evaluating model...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)

print(f"Model Accuracy: {accuracy:.2%}")

# --- 7. Save All Assets ---
print("Saving all model assets...")

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.joblib')

# Save the model
joblib.dump(model, 'model.joblib')

# Save the evaluation metrics
metrics = {
    "accuracy": accuracy,
    "confusion_matrix": cm.tolist(), # Convert numpy array to list for JSON
    "classification_report": report
}

with open('evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("All assets saved successfully!")
print("Files created: vectorizer.joblib, model.joblib, evaluation_metrics.json")