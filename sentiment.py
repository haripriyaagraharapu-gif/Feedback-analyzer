import pandas as pd
from textblob import TextBlob


def analyze_sentiment(text):
    """
    Analyze sentiment of a text using TextBlob.
    Returns dict with label, polarity score, and subjectivity.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'label': 'Neutral', 'polarity': 0.0, 'subjectivity': 0.0}
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            label = 'Positive'
        elif polarity < -0.1:
            label = 'Negative'
        else:
            label = 'Neutral'
        
        return {
            'label': label,
            'polarity': round(polarity, 4),
            'subjectivity': round(subjectivity, 4),
        }
    except Exception:
        return {'label': 'Neutral', 'polarity': 0.0, 'subjectivity': 0.0}


def run_sentiment_analysis(df):
    """
    Apply sentiment analysis to entire DataFrame.
    Adds 'sentiment', 'polarity', 'subjectivity' columns.
    """
    results = df['feedback'].apply(analyze_sentiment)
    
    df = df.copy()
    df['sentiment'] = results.apply(lambda x: x['label'])
    df['polarity'] = results.apply(lambda x: x['polarity'])
    df['subjectivity'] = results.apply(lambda x: x['subjectivity'])
    
    return df


def sentiment_summary(df):
    """Return summary statistics of sentiment distribution."""
    counts = df['sentiment'].value_counts()
    total = len(df)
    
    summary = {}
    for label in ['Positive', 'Negative', 'Neutral']:
        count = counts.get(label, 0)
        summary[label] = {
            'count': int(count),
            'percentage': round((count / total) * 100, 1) if total > 0 else 0.0,
        }
    
    avg_polarity = df['polarity'].mean()
    summary['avg_polarity'] = round(avg_polarity, 4)
    summary['mood'] = 'Generally Positive' if avg_polarity > 0.05 else (
        'Generally Negative' if avg_polarity < -0.05 else 'Mostly Neutral'
    )
    
    return summary
