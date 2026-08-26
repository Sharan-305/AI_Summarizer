import re
import string
import pandas as pd
import nltk

# Ensure NLTK punkt tokenizer is downloaded quietly
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass


def clean_text(text: str) -> str:
    """
    Cleans raw text by stripping HTML tags, removing control characters,
    normalizing spaces, and stripping leading/trailing whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URL links
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Replace zero-width spaces and unusual unicode whitespace
    text = text.replace('\xa0', ' ').replace('\u200b', ' ')
    
    # Normalize multiple whitespace, tabs, and newlines to single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def sentence_split(text: str) -> list:
    """
    Splits long text into individual sentences using NLTK sentence tokenizer,
    with regex fallback if NLTK data is unavailable.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    
    try:
        sentences = nltk.sent_tokenize(cleaned)
    except Exception:
        # Fallback regex sentence tokenizer
        sentences = re.split(r'(?<=[.!?]) +', cleaned)
        
    return [s.strip() for s in sentences if s.strip()]


def preprocess_dataframe(df: pd.DataFrame, text_col: str, summary_col: str = None) -> pd.DataFrame:
    """
    Preprocesses a pandas DataFrame:
    1. Removes missing values (NaN/None/Empty).
    2. Cleans text and summary strings.
    3. Drops duplicate text entries to prevent data leakage.
    """
    df = df.copy()
    
    # Drop rows where text_col is NaN
    df = df.dropna(subset=[text_col])
    if summary_col and summary_col in df.columns:
        df = df.dropna(subset=[summary_col])
        
    # Clean text columns
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    if summary_col and summary_col in df.columns:
        df[summary_col] = df[summary_col].astype(str).apply(clean_text)
        
    # Remove empty string rows
    df = df[df[text_col].str.strip() != ""]
    if summary_col and summary_col in df.columns:
        df = df[df[summary_col].str.strip() != ""]
        
    # Remove duplicates based on input text
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    
    return df


def split_data(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits preprocessed DataFrame into train, validation, and test sets deterministically.
    Ensures strict separation to prevent data leakage.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # Shuffle dataframe with fixed seed
    shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    n_total = len(shuffled_df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_df = shuffled_df.iloc[:n_train].reset_index(drop=True)
    val_df = shuffled_df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = shuffled_df.iloc[n_train + n_val:].reset_index(drop=True)
    
    return train_df, val_df, test_df
