import streamlit as st
import pandas as pd
import base64
from modules import (
    data_collector,
    data_cleaner,
    nlp_processor,
    sentiment,
    topic_classifier,
    visualizer
)

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Feedback Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        color: #94a3b8 !important;
    }
    .stMetric .css-1wivap2 {
        color: #f8fafc !important;
    }
    h1, h2, h3 {
        color: #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b;
        border-bottom: 2px solid #3b82f6;
    }
    .priority-critical { color: #ef4444; font-weight: bold; }
    .priority-high { color: #f59e0b; font-weight: bold; }
    .priority-medium { color: #10b981; font-weight: bold; }
    .priority-low { color: #94a3b8; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# ─── Data Pipeline ────────────────────────────────────────────────────────────
@st.cache_data
def process_data(df):
    """Run the complete NLP and analysis pipeline on the dataframe."""
    # 1. Clean Data
    df_clean, clean_stats = data_cleaner.clean_dataframe(df)
    
    # 2. Sentiment Analysis
    df_sentiment = sentiment.run_sentiment_analysis(df_clean)
    
    # 3. Topic Classification
    df_final = topic_classifier.run_topic_classification(df_sentiment)
    
    return df_final, clean_stats


# ─── Sidebar: Data Input ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("Architecture.png", use_container_width=True) if "Architecture.png" in [f for f in __import__("os").listdir(".")] else None
    st.title("Data Input")
    st.write("Upload feedback data or generate a sample dataset.")
    
    input_method = st.radio("Choose Input Method", ["Generate Sample Data", "Upload CSV"])
    
    df_raw = None
    
    if input_method == "Generate Sample Data":
        n_reviews = st.slider("App Reviews", 0, 50, 20)
        n_tickets = st.slider("Support Tickets", 0, 50, 20)
        n_surveys = st.slider("Surveys", 0, 50, 20)
        
        if st.button("Generate & Process Data", type="primary"):
            df_raw = data_collector.generate_sample_data(n_reviews, n_tickets, n_surveys)
            st.success(f"Generated {len(df_raw)} records successfully!")
            
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            # Save uploaded file to a temporary location
            temp_path = "temp_uploaded.csv"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                df_raw = data_collector.load_from_csv(temp_path)
                st.success(f"Loaded {len(df_raw)} records successfully!")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

# ─── Main Application ─────────────────────────────────────────────────────────
st.title("🎯 Smart Feedback Analyzer")
st.markdown("Analyze customer feedback using NLP to extract sentiment, topics, and actionable insights.")

if df_raw is not None and not df_raw.empty:
    with st.spinner("Processing NLP Pipeline (Cleaning, Sentiment, Topics)..."):
        df_processed, cleaning_stats = process_data(df_raw)
        
        # Calculate Summaries
        sent_summary = sentiment.sentiment_summary(df_processed)
        topic_sum = topic_classifier.topic_summary(df_processed)
        priority_actions = topic_classifier.get_priority_actions(topic_sum)
        kw_df = nlp_processor.keyword_frequency_df(df_processed['tokens'], top_n=20)
        
    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard Overview", "💡 Actionable Insights", "📋 Raw Data & Processing"])
    
    with tab1:
        # Top Level Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Feedback Processed", len(df_processed), f"-{cleaning_stats.get('duplicates_removed', 0)} duplicates")
        col2.metric("Overall Mood", sent_summary['mood'], f"{sent_summary.get('Positive', {}).get('percentage', 0)}% Positive")
        col3.metric("Average Polarity", f"{sent_summary['avg_polarity']:.2f}", "-1 to 1 Scale")
        
        top_topic = priority_actions.iloc[0]['topic'] if not priority_actions.empty else "N/A"
        top_topic_count = priority_actions.iloc[0]['total'] if not priority_actions.empty else 0
        col4.metric("Top Issue Area", top_topic, f"{top_topic_count} mentions")
        
        st.markdown("---")
        
        # Row 1 Charts
        r1c1, r1c2 = st.columns([2, 1])
        with r1c1:
            st.plotly_chart(visualizer.feedback_volume_chart(df_processed), use_container_width=True)
        with r1c2:
            st.plotly_chart(visualizer.sentiment_pie_chart(sent_summary), use_container_width=True)
            
        # Row 2 Charts
        r2c1, r2c2 = st.columns([2, 1])
        with r2c1:
            st.plotly_chart(visualizer.topic_bar_chart(topic_sum), use_container_width=True)
        with r2c2:
            st.plotly_chart(visualizer.source_breakdown_chart(df_processed), use_container_width=True)
            
        # Row 3 Charts & Wordcloud
        r3c1, r3c2, r3c3 = st.columns([1.5, 1.5, 1])
        with r3c1:
            st.plotly_chart(visualizer.keyword_bar_chart(kw_df), use_container_width=True)
        with r3c2:
            st.plotly_chart(visualizer.polarity_distribution_chart(df_processed), use_container_width=True)
        with r3c3:
            st.markdown("<h3 style='text-align: center; color: #e2e8f0; font-size: 16px; font-weight: normal;'>Topic Word Cloud</h3>", unsafe_allow_html=True)
            wc_b64 = visualizer.generate_wordcloud(df_processed['tokens'])
            if wc_b64:
                st.markdown(f'<img src="data:image/png;base64,{wc_b64}" width="100%">', unsafe_allow_html=True)
            else:
                st.info("Not enough data for Word Cloud.")

    with tab2:
        st.subheader("High-Priority Action Items")
        st.markdown("AI-driven recommendations based on topic frequency and negative sentiment concentration.")
        
        # Display Action Items
        for idx, row in priority_actions.iterrows():
            with st.expander(f"{row['priority']} - {row['topic']} ({row['total']} mentions)", expanded=(idx < 2)):
                st.markdown(f"**Recommended Action:** {row['recommended_action']}")
                st.progress(row['negative_pct'] / 100, text=f"Negativity Rate: {row['negative_pct']}%")
                st.caption(f"Breakdown: {row['Positive']} Positive | {row['Neutral']} Neutral | {row['Negative']} Negative")
                
                # Show some sample negative feedback for this topic
                neg_samples = df_processed[(df_processed['topic'] == row['topic']) & (df_processed['sentiment'] == 'Negative')].head(3)
                if not neg_samples.empty:
                    st.markdown("**Sample Negative Feedback:**")
                    for _, s_row in neg_samples.iterrows():
                        st.markdown(f"> *\"{s_row['feedback']}\"*")
                else:
                    st.markdown("*(No explicit negative feedback samples found for this topic.)*")

    with tab3:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("Data Cleaning Stats")
            st.json(cleaning_stats)
            
            st.markdown("### Export")
            csv = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Processed CSV",
                data=csv,
                file_name='processed_feedback.csv',
                mime='text/csv',
            )
            
        with col2:
            st.subheader("Processed DataFrame")
            display_cols = ['id', 'source', 'feedback', 'sentiment', 'topic', 'polarity']
            st.dataframe(df_processed[display_cols], use_container_width=True, height=500)

else:
    st.info("👈 Please generate sample data or upload a CSV file from the sidebar to begin analysis.")
    
    # Show Architecture Diagram on empty state
    try:
        st.image("Architecture.png", caption="System Architecture", width=800)
    except:
        pass
