import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import nltk
from langdetect import detect
from tqdm import tqdm

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Language code to full name mapping
LANG_MAP = {
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'nl': 'dutch',
    'ru': 'russian',
    'ar': 'arabic',
    'ja': 'japanese',
    'ko': 'korean',
    'zh-cn': 'chinese',
    'zh-tw': 'chinese'
}

def load_data(file_path):
    """Load the data from a CSV file."""
    return pd.read_csv(file_path)

def detect_language(text):
    """Detect the language of the text."""
    try:
        return detect(text)
    except:
        return 'unknown'

def get_full_lang_name(lang_code):
    """Convert language code to full name."""
    return LANG_MAP.get(lang_code, 'english')  # Default to English if not found

def preprocess_text(text, language):
    """Preprocess the text data."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    full_lang_name = get_full_lang_name(language)
    if full_lang_name in stopwords.fileids():
        stop_words = set(stopwords.words(full_lang_name))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    try:
        stemmer = SnowballStemmer(full_lang_name)
        tokens = [stemmer.stem(token) for token in tokens]
    except ValueError:
        # If stemmer is not available for the language, skip stemming
        pass
    
    return ' '.join(tokens)

def determine_stance(text):
    """Determine the stance based on the text content."""
    positive_keywords = ['support', 'agree', 'real', 'urgent', 'action', 'believe', 'serious', 'threat', 'danger']
    negative_keywords = ['hoax', 'fake', 'exaggerated', 'myth', 'scam', 'conspiracy', 'overblown', 'alarmist']
    
    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in positive_keywords):
        return 0  # Positive stance
    elif any(keyword in text_lower for keyword in negative_keywords):
        return 1  # Negative stance
    else:
        return 2  # Neutral stance

def prepare_data(df):
    """Prepare the data for modeling."""
    # Combine title and body
    df['text'] = df['title'] + ' ' + df['body'].fillna('')
    
    # Detect language if not provided
    if 'language' not in df.columns:
        tqdm.pandas(desc="Detecting languages")
        df['language'] = df['text'].progress_apply(detect_language)
    
    # Preprocess text
    tqdm.pandas(desc="Preprocessing text")
    df['processed_text'] = df.progress_apply(lambda row: preprocess_text(row['text'], row['language']), axis=1)
    
    # Determine stance
    tqdm.pandas(desc="Determining stance")
    df['stance'] = df['text'].progress_apply(determine_stance)
    
    return df

def split_data(df, test_size=0.2, val_size=0.1):
    """Split the data into train, validation, and test sets."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df['language'], random_state=42)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), stratify=train_val['language'], random_state=42)
    return train, val, test

def main():
    # Load data
    data = load_data('data/raw/reddit_climate_data_20240831_010210.csv')  # Replace with your actual filename
    
    # Prepare data
    prepared_data = prepare_data(data)
    
    # Split data
    train, val, test = split_data(prepared_data)
    
    # Save processed data
    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    
    print(f"Processed data saved. Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

if __name__ == "__main__":
    main()