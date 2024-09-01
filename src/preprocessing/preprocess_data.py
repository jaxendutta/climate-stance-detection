import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load the data
# Adjust the file path as necessary
df = pd.read_csv('../../data/raw/reddit_climate_data_20240831_010210.csv')

# Display basic information about the dataset
print(df.info())
print(df['language'].value_counts())
print(df['simple_stance'].value_counts())

# Function for text preprocessing
def preprocess_text(text, language):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'\W|\d', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words(language))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = SnowballStemmer(language)
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing to the 'text' column
df['processed_text'] = df.apply(lambda row: preprocess_text(row['text'], row['language']), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['simple_stance'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a simple Logistic Regression model as baseline
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Print classification report
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()