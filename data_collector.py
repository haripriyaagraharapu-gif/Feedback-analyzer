import pandas as pd
import random
from datetime import datetime, timedelta

# Sample feedback data for each source
SAMPLE_APP_REVIEWS = [
    "The app crashes every time I try to make a payment. Very frustrating!",
    "Love the new UI design, it's so much cleaner and intuitive.",
    "Payment gateway keeps failing. I've lost money twice because of this.",
    "The loading speed has improved drastically. Great update!",
    "Can't log in after the latest update. Completely broken!",
    "Notification system is excellent. I always get timely alerts.",
    "Why is the UI so confusing? Nothing makes sense.",
    "Battery drain is insane with this app. Needs optimization.",
    "Customer support was incredibly helpful and fast to respond.",
    "App is great but needs dark mode support desperately.",
    "Performance is buttery smooth. Best app in its category!",
    "The checkout process has too many steps. Simplify please.",
    "Frequent crashes make this unusable. Fix it!",
    "I love how personalized the recommendations are. Spot on!",
    "Payment fails 3 out of 5 times. Terrible experience.",
    "Clean interface and easy to navigate. 5 stars!",
    "The search functionality is broken. Can't find anything.",
    "Amazing app, just needs a few bug fixes.",
    "Worst app update ever. Everything stopped working.",
    "Fast, reliable, and beautifully designed. Highly recommend!",
]

SAMPLE_SUPPORT_TICKETS = [
    "I was charged twice for my order. Please refund immediately.",
    "Account locked after wrong password. Need help resetting.",
    "The product I received is damaged. Requesting replacement.",
    "I cannot access my account settings after the recent update.",
    "Payment declined repeatedly despite valid card details.",
    "App keeps logging me out. Very inconvenient.",
    "I need to cancel my subscription but can't find the option.",
    "Data sync is not working between my devices.",
    "The refund process took too long. Very disappointed.",
    "Can you add a feature to export my data as PDF?",
    "Login page is broken on mobile devices.",
    "I'm getting error 500 every time I try to view my history.",
    "The coupon code is not being applied at checkout.",
    "Please improve the search filters. Current ones are useless.",
    "My notification preferences keep resetting.",
    "Cannot update my billing information. The form is broken.",
    "Excellent support team! Resolved my issue in minutes.",
    "Why was my account suspended without any warning?",
    "The new update removed a feature I used daily.",
    "Integration with third-party apps is not working.",
]

SAMPLE_SURVEYS = [
    "Overall satisfied but the performance could be better on older devices.",
    "The new features are great but the tutorial is inadequate.",
    "I wish there were more payment options available.",
    "User experience is top-notch. Everything is exactly where I expect it.",
    "The app is good but the price is too high for what it offers.",
    "I've been using this for 2 years and it keeps getting better.",
    "Customer service response time needs significant improvement.",
    "The analytics feature is incredibly useful for my business.",
    "Onboarding process is very smooth for new users.",
    "I'd rate the UI 9/10 but data export options are lacking.",
    "Please add multi-language support for global users.",
    "The offline mode is a game changer for remote areas.",
    "Too many ads interrupt the user experience.",
    "Would love to see a desktop version with more features.",
    "The collaboration tools are exactly what our team needed.",
    "Performance issues on iOS persist even after updates.",
    "Excellent product overall, minor bugs are expected.",
    "The dashboard is powerful but initially overwhelming.",
    "I appreciate how quickly bugs are patched after reporting.",
    "The mobile app needs a complete redesign. Feels outdated.",
]


def generate_sample_data(n_reviews=20, n_tickets=20, n_surveys=20):
    """Generate sample feedback data from multiple sources."""
    all_data = []
    
    base_date = datetime.now() - timedelta(days=30)
    
    # App reviews
    for i, text in enumerate(SAMPLE_APP_REVIEWS[:n_reviews]):
        all_data.append({
            'id': f'APP-{i+1:03d}',
            'source': 'App Store',
            'feedback': text,
            'date': (base_date + timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
            'rating': random.choice([1, 2, 3, 4, 5]),
        })
    
    # Support tickets
    for i, text in enumerate(SAMPLE_SUPPORT_TICKETS[:n_tickets]):
        all_data.append({
            'id': f'TKT-{i+1:03d}',
            'source': 'Support Ticket',
            'feedback': text,
            'date': (base_date + timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
            'rating': None,
        })
    
    # Surveys
    for i, text in enumerate(SAMPLE_SURVEYS[:n_surveys]):
        all_data.append({
            'id': f'SRV-{i+1:03d}',
            'source': 'Survey',
            'feedback': text,
            'date': (base_date + timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
            'rating': random.choice([3, 4, 5]),
        })
    
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def load_from_csv(filepath):
    """Load feedback from a user-uploaded CSV file."""
    try:
        df = pd.read_csv(filepath)
        # Normalize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower in ['feedback', 'review', 'comment', 'text', 'message', 'description']:
                col_map[col] = 'feedback'
            elif lower in ['source', 'platform', 'channel', 'type']:
                col_map[col] = 'source'
            elif lower in ['date', 'created_at', 'timestamp', 'time']:
                col_map[col] = 'date'
            elif lower in ['id', 'ticket_id', 'review_id']:
                col_map[col] = 'id'
            elif lower in ['rating', 'score', 'stars']:
                col_map[col] = 'rating'
        
        df = df.rename(columns=col_map)
        
        if 'feedback' not in df.columns:
            # Use the first text-like column
            text_cols = df.select_dtypes(include='object').columns
            if len(text_cols) > 0:
                df = df.rename(columns={text_cols[0]: 'feedback'})
        
        if 'source' not in df.columns:
            df['source'] = 'Uploaded CSV'
        
        if 'date' not in df.columns:
            df['date'] = datetime.now().strftime('%Y-%m-%d')
        
        if 'id' not in df.columns:
            df['id'] = [f'CSV-{i+1:03d}' for i in range(len(df))]
        
        if 'rating' not in df.columns:
            df['rating'] = None
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['feedback'])
        df = df[df['feedback'].str.strip() != '']
        
        return df[['id', 'source', 'feedback', 'date', 'rating']].reset_index(drop=True)
    
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {str(e)}")


def save_to_csv(df, filepath):
    """Save processed dataframe to CSV."""
    df.to_csv(filepath, index=False)
