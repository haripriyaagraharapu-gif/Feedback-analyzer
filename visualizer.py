import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# ─── Color Palette ────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    'Positive': '#10b981',   # emerald
    'Neutral':  '#6366f1',   # indigo
    'Negative': '#ef4444',   # red
}

TOPIC_COLORS = [
    '#6366f1', '#8b5cf6', '#ec4899', '#f59e0b',
    '#10b981', '#06b6d4', '#f97316', '#84cc16',
]

CHART_BG = 'rgba(0,0,0,0)'
FONT_COLOR = '#e2e8f0'
GRID_COLOR = '#334155'

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=FONT_COLOR, family='Inter, sans-serif'),
    margin=dict(l=20, r=20, t=40, b=20),
)


def sentiment_pie_chart(sentiment_counts: dict):
    """Donut chart showing sentiment distribution."""
    labels = list(sentiment_counts.keys())
    values = [sentiment_counts[l]['count'] for l in labels]
    colors = [SENTIMENT_COLORS.get(l, '#94a3b8') for l in labels]
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#1e293b', width=2)),
        textinfo='label+percent',
        textfont=dict(size=13, color=FONT_COLOR),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>',
    ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='Sentiment Distribution', font=dict(size=16, color=FONT_COLOR), x=0.5),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15,
                    font=dict(color=FONT_COLOR)),
        height=340,
    )
    return fig


def topic_bar_chart(topic_df: pd.DataFrame):
    """Grouped bar chart showing topics with sentiment breakdown."""
    if topic_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    sentiments = [('Negative', '#ef4444'), ('Neutral', '#6366f1'), ('Positive', '#10b981')]
    for sentiment, color in sentiments:
        if sentiment in topic_df.columns:
            fig.add_trace(go.Bar(
                name=sentiment,
                x=topic_df['topic'],
                y=topic_df[sentiment],
                marker_color=color,
                hovertemplate=f'<b>%{{x}}</b><br>{sentiment}: %{{y}}<extra></extra>',
            ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='Issues by Topic & Sentiment', font=dict(size=16, color=FONT_COLOR), x=0.5),
        barmode='stack',
        xaxis=dict(tickangle=-20, gridcolor=GRID_COLOR, showgrid=False),
        yaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, font=dict(color=FONT_COLOR)),
        height=380,
    )
    return fig


def keyword_bar_chart(keyword_df: pd.DataFrame):
    """Horizontal bar chart for top keywords."""
    if keyword_df.empty:
        return go.Figure()
    
    top = keyword_df.head(15).sort_values('count')
    
    fig = go.Figure(go.Bar(
        x=top['count'],
        y=top['keyword'],
        orientation='h',
        marker=dict(
            color=top['count'],
            colorscale=[[0, '#4f46e5'], [0.5, '#7c3aed'], [1, '#ec4899']],
            showscale=False,
        ),
        text=top['count'],
        textposition='outside',
        textfont=dict(color=FONT_COLOR, size=11),
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>',
    ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='Top Keywords', font=dict(size=16, color=FONT_COLOR), x=0.5),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
        yaxis=dict(showgrid=False),
        height=400,
    )
    return fig


def feedback_volume_chart(df: pd.DataFrame):
    """Line chart showing feedback volume over time by source."""
    if 'date' not in df.columns or df['date'].isna().all():
        return go.Figure()
    
    df_time = df.copy()
    df_time['date'] = pd.to_datetime(df_time['date'], errors='coerce')
    df_time = df_time.dropna(subset=['date'])
    df_time['week'] = df_time['date'].dt.to_period('W').apply(lambda r: r.start_time)
    
    grouped = df_time.groupby(['week', 'source']).size().reset_index(name='count')
    
    sources = grouped['source'].unique()
    source_colors = dict(zip(sources, TOPIC_COLORS[:len(sources)]))
    
    fig = go.Figure()
    for source in sources:
        subset = grouped[grouped['source'] == source]
        fig.add_trace(go.Scatter(
            x=subset['week'],
            y=subset['count'],
            name=source,
            mode='lines+markers',
            line=dict(color=source_colors.get(source, '#6366f1'), width=2.5),
            marker=dict(size=6),
            hovertemplate=f'<b>{source}</b><br>Week: %{{x|%b %d}}<br>Count: %{{y}}<extra></extra>',
        ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='Feedback Volume Over Time', font=dict(size=16, color=FONT_COLOR), x=0.5),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, tickformat='%b %d'),
        yaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, font=dict(color=FONT_COLOR)),
        height=320,
    )
    return fig


def polarity_distribution_chart(df: pd.DataFrame):
    """Histogram of polarity scores."""
    if 'polarity' not in df.columns:
        return go.Figure()
    
    fig = go.Figure(go.Histogram(
        x=df['polarity'],
        nbinsx=20,
        marker=dict(
            color=df['polarity'],
            colorscale=[[0, '#ef4444'], [0.5, '#6366f1'], [1, '#10b981']],
            showscale=False,
            line=dict(color='#1e293b', width=0.5),
        ),
        hovertemplate='Polarity: %{x:.2f}<br>Count: %{y}<extra></extra>',
    ))
    
    fig.add_vline(x=0, line_dash='dash', line_color='#94a3b8', line_width=1.5)
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='Polarity Score Distribution', font=dict(size=16, color=FONT_COLOR), x=0.5),
        xaxis=dict(title='Polarity (-1 = Negative, +1 = Positive)', gridcolor=GRID_COLOR),
        yaxis=dict(title='Count', gridcolor=GRID_COLOR),
        height=300,
    )
    return fig


def source_breakdown_chart(df: pd.DataFrame):
    """Pie chart for feedback source distribution."""
    source_counts = df['source'].value_counts()
    
    fig = go.Figure(go.Pie(
        labels=source_counts.index.tolist(),
        values=source_counts.values.tolist(),
        hole=0.45,
        marker=dict(
            colors=TOPIC_COLORS[:len(source_counts)],
            line=dict(color='#1e293b', width=2),
        ),
        textinfo='label+percent',
        textfont=dict(size=12, color=FONT_COLOR),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>',
    ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='Feedback by Source', font=dict(size=16, color=FONT_COLOR), x=0.5),
        showlegend=False,
        height=300,
    )
    return fig


def generate_wordcloud(tokens_series):
    """Generate a word cloud image from token series, returned as base64 PNG."""
    all_text = []
    for tokens in tokens_series:
        if isinstance(tokens, list):
            all_text.extend(tokens)
    
    if not all_text:
        return None
    
    text_joined = ' '.join(all_text)
    
    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color=None,
            mode='RGBA',
            colormap='cool',
            max_words=80,
            prefer_horizontal=0.8,
            font_path=None,
        ).generate(text_joined)
        
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='none')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        fig.patch.set_alpha(0)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        return img_base64
    except Exception:
        return None
