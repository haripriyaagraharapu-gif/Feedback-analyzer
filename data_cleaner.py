import re
import string
import nltk
import pandas as pd

# Download required NLTK data
def ensure_nltk_data():
    packages = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger']
    for pkg in packages:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

ensure_nltk_data()

try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                      'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her',
                      'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who',
                      'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
                      'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
                      'does', 'did', 'will', 'would', 'could', 'should', 'may',
                      'the', 'a', 'an', 'and', 'but', 'or', 'nor', 'for', 'yet',
                      'so', 'to', 'of', 'in', 'on', 'at', 'by', 'from', 'with',
                      'about', 'as', 'into', 'through', 'during', 'before', 'after',
                      'very', 'just', 'can', 'not', 'up', 'out', 'if'])


def clean_text(text):
    """Clean a single text string."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep apostrophes for contractions
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text):
    """Remove common stopwords from text."""
    words = text.split()
    filtered = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(filtered)


def clean_dataframe(df):
    """
    Apply full cleaning pipeline to the feedback DataFrame.
    Returns cleaned DataFrame with stats.
    """
    stats = {}
    original_count = len(df)
    stats['original_count'] = original_count
    
    # 1. Drop rows with empty/null feedback
    df = df.dropna(subset=['feedback'])
    df = df[df['feedback'].str.strip() != '']
    stats['after_null_removal'] = len(df)
    
    # 2. Remove exact duplicates
    df = df.drop_duplicates(subset=['feedback'], keep='first')
    stats['after_dedup'] = len(df)
    stats['duplicates_removed'] = stats['after_null_removal'] - stats['after_dedup']
    
    # 3. Clean text (store in new column, keep original)
    df = df.copy()
    df['cleaned_text'] = df['feedback'].apply(clean_text)
    
    # 4. Remove very short texts after cleaning (less than 5 chars)
    df = df[df['cleaned_text'].str.len() >= 5]
    stats['after_short_removal'] = len(df)
    stats['short_removed'] = stats['after_dedup'] - stats['after_short_removal']
    
    # 5. Create tokens (for keyword analysis)
    df['tokens'] = df['cleaned_text'].apply(
        lambda t: [w for w in t.split() if w not in STOP_WORDS and len(w) > 2]
    )
    
    stats['final_count'] = len(df)
    stats['total_removed'] = original_count - stats['final_count']
    stats['removal_pct'] = round((stats['total_removed'] / original_count) * 100, 1) if original_count > 0 else 0
    
    return df.reset_index(drop=True), stats
