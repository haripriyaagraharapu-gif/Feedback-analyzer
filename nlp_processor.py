from collections import Counter
import pandas as pd

# Common words to exclude from keyword analysis (beyond stopwords)
EXCLUDE_WORDS = {
    'app', 'use', 'used', 'using', 'get', 'got', 'go', 'need', 'needs',
    'would', 'could', 'should', 'also', 'like', 'just', 'really', 'make',
    'even', 'one', 'two', 'first', 'time', 'times', 'every', 'after',
    'still', 'keep', 'keeps', 'thing', 'things', 'way', 'says', 'said',
    'much', 'many', 'good', 'bad', 'great', 'new', 'old', 'best', 'worst',
    'work', 'works', 'working', 'worked', "doesn't", "can't", "won't",
    "isn't", "it's", "i'm", "i've", "i'd", "they're", "don't"
}


def extract_keywords(tokens_series, top_n=20):
    """
    Extract top keywords from a series of token lists.
    Returns list of (word, count) tuples.
    """
    all_tokens = []
    for token_list in tokens_series:
        if isinstance(token_list, list):
            all_tokens.extend([t for t in token_list if t not in EXCLUDE_WORDS and len(t) > 3])
    
    counter = Counter(all_tokens)
    return counter.most_common(top_n)


def get_bigrams(tokens_series, top_n=10):
    """Extract top bigrams (two-word phrases) from token series."""
    all_bigrams = []
    for token_list in tokens_series:
        if isinstance(token_list, list) and len(token_list) >= 2:
            bigrams = [(token_list[i] + ' ' + token_list[i+1])
                       for i in range(len(token_list) - 1)
                       if token_list[i] not in EXCLUDE_WORDS
                       and token_list[i+1] not in EXCLUDE_WORDS]
            all_bigrams.extend(bigrams)
    
    counter = Counter(all_bigrams)
    return counter.most_common(top_n)


def keyword_frequency_df(tokens_series, top_n=20):
    """Return a DataFrame of keyword frequencies."""
    keywords = extract_keywords(tokens_series, top_n)
    return pd.DataFrame(keywords, columns=['keyword', 'count'])
