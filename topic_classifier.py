import re
import pandas as pd

# Topic keyword mapping — each topic has a set of trigger keywords
TOPIC_PATTERNS = {
    'Payment Issues': [
        'payment', 'pay', 'charge', 'charged', 'refund', 'billing', 'bill',
        'transaction', 'checkout', 'coupon', 'subscription', 'fee', 'cost',
        'purchase', 'money', 'card', 'declined', 'credit', 'debit', 'invoice',
    ],
    'Login & Access': [
        'login', 'log in', 'signin', 'sign in', 'logout', 'password', 'account',
        'locked', 'access', 'authentication', 'credentials', 'reset', 'session',
        'suspended', 'unauthorized', 'register', 'signup', 'sign up',
    ],
    'Performance': [
        'crash', 'crashes', 'slow', 'speed', 'lag', 'freeze', 'frozen', 'hang',
        'loading', 'load', 'performance', 'fast', 'quick', 'battery', 'memory',
        'optimize', 'optimization', 'fps', 'respond', 'response', 'timeout',
    ],
    'UI/UX': [
        'ui', 'ux', 'interface', 'design', 'layout', 'button', 'screen',
        'navigation', 'menu', 'confusing', 'intuitive', 'user experience',
        'dashboard', 'display', 'visual', 'theme', 'dark mode', 'color',
        'font', 'icon', 'responsive', 'mobile', 'desktop', 'outdated',
    ],
    'Features': [
        'feature', 'features', 'option', 'add', 'missing', 'support',
        'export', 'import', 'filter', 'search', 'notification', 'alert',
        'language', 'offline', 'sync', 'integration', 'analytics', 'report',
        'pdf', 'csv', 'download', 'upload', 'multi', 'collaboration',
    ],
    'Customer Support': [
        'support', 'customer service', 'help', 'assist', 'response', 'team',
        'resolved', 'resolve', 'contact', 'reply', 'ticket', 'agent',
        'wait', 'waiting', 'time', 'slow response', 'helpful',
    ],
    'Data & Privacy': [
        'data', 'privacy', 'security', 'secure', 'leak', 'breach', 'encrypt',
        'personal', 'information', 'gdpr', 'delete', 'account deletion',
        'tracking', 'ads', 'advertisement', 'permission',
    ],
    'General Feedback': [],  # Catch-all
}


def classify_topic(text):
    """Classify a feedback text into a topic category."""
    if not isinstance(text, str):
        return 'General Feedback'
    
    text_lower = text.lower()
    
    topic_scores = {}
    for topic, keywords in TOPIC_PATTERNS.items():
        if topic == 'General Feedback':
            continue
        score = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
        if score > 0:
            topic_scores[topic] = score
    
    if topic_scores:
        return max(topic_scores, key=topic_scores.get)
    return 'General Feedback'


def run_topic_classification(df):
    """Apply topic classification to entire DataFrame."""
    df = df.copy()
    df['topic'] = df['feedback'].apply(classify_topic)
    return df


def topic_summary(df):
    """Return topic frequency and breakdown by sentiment."""
    if 'topic' not in df.columns or 'sentiment' not in df.columns:
        return pd.DataFrame()
    
    summary = df.groupby(['topic', 'sentiment']).size().unstack(fill_value=0)
    summary['total'] = summary.sum(axis=1)
    
    # Ensure all sentiment columns exist
    for col in ['Positive', 'Negative', 'Neutral']:
        if col not in summary.columns:
            summary[col] = 0
    
    summary = summary[['Positive', 'Negative', 'Neutral', 'total']]
    summary = summary.sort_values('total', ascending=False)
    summary['negative_pct'] = (summary['Negative'] / summary['total'] * 100).round(1)
    
    return summary.reset_index()


def get_priority_actions(topic_df):
    """
    Generate prioritized action items based on topic analysis.
    Priority = negative count * 2 + total count
    """
    if topic_df.empty:
        return pd.DataFrame()
    
    df = topic_df.copy()
    df['priority_score'] = df['Negative'] * 2 + df['total']
    df = df.sort_values('priority_score', ascending=False)
    
    action_map = {
        'Payment Issues': 'Audit and fix payment gateway reliability; test edge cases',
        'Login & Access': 'Review authentication flow; add account recovery options',
        'Performance': 'Profile app performance; optimize critical rendering paths',
        'UI/UX': 'Conduct user testing sessions; redesign problem areas',
        'Features': 'Prioritize top-requested features in next sprint',
        'Customer Support': 'Reduce response time SLA; add self-service help center',
        'Data & Privacy': 'Conduct security audit; update privacy controls',
        'General Feedback': 'Review and tag uncategorized feedback manually',
    }
    
    df['recommended_action'] = df['topic'].map(action_map).fillna('Review and address user concerns')
    
    priority_labels = []
    for score in df['priority_score']:
        if score >= 15:
            priority_labels.append('🔴 Critical')
        elif score >= 8:
            priority_labels.append('🟡 High')
        elif score >= 4:
            priority_labels.append('🟢 Medium')
        else:
            priority_labels.append('⚪ Low')
    
    df['priority'] = priority_labels
    
    return df[['topic', 'total', 'Positive', 'Negative', 'Neutral',
                'negative_pct', 'priority', 'recommended_action']]
