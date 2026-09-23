"""
News Article Bias & Similarity Analyzer
A production-ready web application for analyzing news article bias, sentiment, and credibility.

Author: Senior ML Engineer
Tech Stack: Streamlit, transformers, sentence-transformers, newspaper3k, nltk
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional
import hashlib
import requests

# NLP & ML Libraries
from newspaper import Article
from textblob import TextBlob
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK datasets on first run"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
    except Exception as e:
        st.warning(f"NLTK download warning: {e}")

download_nltk_data()


# ===== MODEL LOADING (CACHED FOR PERFORMANCE) =====

@st.cache_resource
def load_sentiment_model():
    """Load transformer-based sentiment analysis model (cached)"""
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU
    )

@st.cache_resource
def load_sentence_transformer():
    """Load sentence embedding model for similarity analysis (cached)"""
    return SentenceTransformer('all-MiniLM-L6-v2')


# ===== CORE FUNCTIONS: ARTICLE EXTRACTION =====

def extract_article(url: str) -> Dict:
    """
    Extract article content from URL using requests and newspaper3k
    """
    try:
        clean_url = url.strip().strip("'\"")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        response = requests.get(clean_url, headers=headers, timeout=12)
        response.raise_for_status()
        
        article = Article(clean_url)
        article.download(input_html=response.text)
        article.parse()
        
        return {
            'success': True,
            'title': article.title,
            'text': article.text,
            'authors': article.authors,
            'publish_date': article.publish_date,
            'url': clean_url,
            'top_image': article.top_image,
            'word_count': len(article.text.split())
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# ===== CORE FUNCTIONS: SENTIMENT & BIAS ANALYSIS =====

def analyze_sentiment(text: str, sentiment_model) -> Dict:
    """
    Perform sentiment analysis using both TextBlob and transformers
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    words = text.split()
    chunks = [' '.join(words[i:i+100]) for i in range(0, len(words), 100)][:5]
    
    transformer_results = []
    for chunk in chunks:
        if len(chunk.strip()) > 0:
            result = sentiment_model(chunk[:512])[0]
            transformer_results.append(result)
    
    positive_scores = [r['score'] for r in transformer_results if r['label'] == 'POSITIVE']
    negative_scores = [r['score'] for r in transformer_results if r['label'] == 'NEGATIVE']
    
    avg_positive = np.mean(positive_scores) if positive_scores else 0
    avg_negative = np.mean(negative_scores) if negative_scores else 0
    
    if avg_positive > avg_negative:
        overall_label = "POSITIVE"
        confidence = avg_positive
    else:
        overall_label = "NEGATIVE"
        confidence = avg_negative
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'transformer_label': overall_label,
        'transformer_confidence': confidence,
        'positive_score': avg_positive,
        'negative_score': avg_negative
    }


def calculate_bias_metrics(sentiment_data: Dict, text: str) -> Dict:
    """
    Calculate bias indicators from sentiment and text analysis
    """
    emotional_bias = abs(sentiment_data['polarity'])
    subjectivity = sentiment_data['subjectivity']
    
    strong_words = [
        'extremely', 'absolutely', 'completely', 'totally', 'utterly',
        'devastating', 'incredible', 'amazing', 'terrible', 'horrible',
        'fantastic', 'awful', 'brilliant', 'shocking', 'outrageous'
    ]
    text_lower = text.lower()
    intensity_count = sum(text_lower.count(word) for word in strong_words)
    word_count = len(text.split())
    intensity_ratio = (intensity_count / word_count) * 100 if word_count > 0 else 0
    
    return {
        'emotional_bias': emotional_bias,
        'subjectivity': subjectivity,
        'intensity_ratio': intensity_ratio,
        'bias_category': categorize_bias(subjectivity, emotional_bias)
    }


def categorize_bias(subjectivity: float, emotional_bias: float) -> str:
    """Categorize article bias level"""
    if subjectivity > 0.6 or emotional_bias > 0.5:
        return "High Bias (Opinion-Heavy)"
    elif subjectivity > 0.4 or emotional_bias > 0.3:
        return "Moderate Bias"
    else:
        return "Low Bias (Fact-Based)"


def calculate_credibility_score(bias_metrics: Dict, sentiment_data: Dict) -> float:
    """
    Calculate credibility score (0-100) based on objectivity and balance
    """
    score = 100.0
    score -= (bias_metrics['subjectivity'] * 40)
    score -= (bias_metrics['emotional_bias'] * 30)
    score -= min(bias_metrics['intensity_ratio'] * 2, 20)
    score -= (abs(sentiment_data['polarity']) * 10)
    return max(0, min(100, score))


# ===== CORE FUNCTIONS: SIMILARITY ANALYSIS =====

@st.cache_data
def get_reference_articles() -> List[Dict]:
    return [
        {
            'title': 'Climate Change: Scientists Warn of Tipping Points',
            'text': 'Recent climate research indicates that global temperatures are approaching critical thresholds. Scientists from leading institutions have published findings showing accelerated ice melt in polar regions. The data suggests that current emission trends could lead to irreversible changes in weather patterns.',
            'source': 'Science News'
        },
        {
            'title': 'Economic Growth Slows in Major Markets',
            'text': 'Economic indicators show a slowdown in growth across developed economies. Central banks are monitoring inflation rates closely while considering policy adjustments. Manufacturing output has decreased, and consumer spending shows signs of caution.',
            'source': 'Financial Times'
        },
        {
            'title': 'Breakthrough in Renewable Energy Technology',
            'text': 'Researchers have developed a new solar panel design that significantly improves efficiency. The innovation uses advanced materials to capture a broader spectrum of light. Early testing shows a 40% improvement over conventional panels, potentially reducing costs for widespread adoption.',
            'source': 'Tech Review'
        },
        {
            'title': 'Healthcare Access Remains Major Challenge',
            'text': 'Despite advances in medical technology, millions still lack access to basic healthcare services. Rural areas face particular difficulties with doctor shortages and infrastructure limitations. Policy makers debate solutions ranging from telemedicine to incentive programs for healthcare workers.',
            'source': 'Health Policy Today'
        },
        {
            'title': 'Artificial Intelligence Transforms Industries',
            'text': 'AI applications are rapidly expanding across sectors from finance to manufacturing. Machine learning algorithms now handle tasks previously requiring human expertise. Companies report productivity gains but also raise concerns about workforce displacement and the need for retraining programs.',
            'source': 'Technology Weekly'
        }
    ]


def compute_similarity(article_text: str, embedding_model) -> List[Dict]:
    reference_articles = get_reference_articles()
    all_texts = [article_text] + [ref['text'] for ref in reference_articles]
    embeddings = embedding_model.encode(all_texts)
    
    input_embedding = embeddings[0].reshape(1, -1)
    reference_embeddings = embeddings[1:]
    
    similarities = cosine_similarity(input_embedding, reference_embeddings)[0]
    
    results = []
    for i, ref in enumerate(reference_articles):
        results.append({
            'title': ref['title'],
            'source': ref['source'],
            'similarity_score': float(similarities[i]),
            'similarity_percent': float(similarities[i] * 100)
        })
    
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results


# ===== UI HELPER FUNCTIONS =====

def display_metrics_cards(sentiment_data: Dict, bias_metrics: Dict, credibility: float):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Sentiment",
            sentiment_data['transformer_label'],
            f"{sentiment_data['transformer_confidence']:.2%} confidence"
        )
    
    with col2:
        st.metric(
            "Subjectivity",
            f"{bias_metrics['subjectivity']:.2%}",
            "Opinion vs Fact"
        )
    
    with col3:
        st.metric(
            "Emotional Bias",
            f"{bias_metrics['emotional_bias']:.2%}",
            bias_metrics['bias_category']
        )
    
    with col4:
        st.metric(
            "Credibility Score",
            f"{credibility:.0f}/100",
            "Objectivity Rating"
        )


def display_sentiment_chart(sentiment_data: Dict):
    chart_data = pd.DataFrame({
        'Sentiment Type': ['Positive', 'Negative'],
        'Score': [
            sentiment_data['positive_score'],
            sentiment_data['negative_score']
        ]
    })
    st.bar_chart(chart_data.set_index('Sentiment Type'))


def display_bias_breakdown(bias_metrics: Dict, sentiment_data: Dict):
    breakdown_data = pd.DataFrame({
        'Bias Component': [
            'Subjectivity',
            'Emotional Language',
            'Sentiment Polarity',
            'Language Intensity'
        ],
        'Score': [
            bias_metrics['subjectivity'] * 100,
            bias_metrics['emotional_bias'] * 100,
            abs(sentiment_data['polarity']) * 100,
            min(bias_metrics['intensity_ratio'], 100)
        ]
    })
    
    st.dataframe(
        breakdown_data.style.format({'Score': '{:.1f}'}),
        use_container_width=True,
        hide_index=True
    )


def display_similarity_results(similarity_results: List[Dict]):
    st.subheader("📊 Similar Articles")
    st.caption("Semantic similarity based on content and meaning")
    
    for i, result in enumerate(similarity_results[:3], 1):
        with st.expander(
            f"#{i} - {result['title']} ({result['similarity_percent']:.1f}% similar)",
            expanded=(i == 1)
        ):
            st.write(f"**Source:** {result['source']}")
            st.progress(result['similarity_score'])
            
            if result['similarity_percent'] > 70:
                st.info("🔵 Very similar topic and framing")
            elif result['similarity_percent'] > 50:
                st.info("🟢 Moderately similar themes")
            else:
                st.info("🟡 Some topical overlap")
    
    with st.expander("View All Similarity Scores"):
        df = pd.DataFrame(similarity_results)
        st.dataframe(
            df[['title', 'source', 'similarity_percent']].style.format({
                'similarity_percent': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )


# ===== MAIN APPLICATION =====

def main():
    st.set_page_config(
        page_title="News Bias Analyzer",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("📰 News Article Bias & Similarity Analyzer")
    st.markdown("""
    Analyze news articles for **sentiment**, **bias**, **credibility**, and **semantic similarity** using advanced NLP models.
    """)
    
    st.divider()
    
    st.subheader("🔗 Input Article URL")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input(
            "Enter news article URL",
            placeholder="https://example.com/news/article",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
    
    st.divider()
    
    if analyze_button and url:
        url_pattern = re.compile(r'https?://.+')
        if not url_pattern.match(url):
            st.error("❌ Please enter a valid URL starting with http:// or https://")
            return
        
        with st.spinner("🔄 Extracting and analyzing article... This may take 10-30 seconds."):
            article_data = extract_article(url)
            
            if not article_data['success']:
                st.error(f"❌ Failed to extract article: {article_data['error']}")
                st.info("Ensure the URL is publicly accessible without a hard paywall.")
                return
            
            sentiment_model = load_sentiment_model()
            embedding_model = load_sentence_transformer()
            
            sentiment_data = analyze_sentiment(article_data['text'], sentiment_model)
            bias_metrics = calculate_bias_metrics(sentiment_data, article_data['text'])
            credibility = calculate_credibility_score(bias_metrics, sentiment_data)
            similarity_results = compute_similarity(article_data['text'], embedding_model)
        
        st.success("✅ Analysis Complete!")
        
        st.subheader("📄 Article Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.write(f"**Title:** {article_data['title'][:100]}...")
        with info_col2:
            st.write(f"**Word Count:** {article_data['word_count']:,}")
        with info_col3:
            if article_data['publish_date']:
                st.write(f"**Published:** {article_data['publish_date'].strftime('%Y-%m-%d')}")
            else:
                st.write("**Published:** Unknown")
        
        st.divider()
        
        st.subheader("🎯 Key Metrics")
        display_metrics_cards(sentiment_data, bias_metrics, credibility)
        
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["📊 Sentiment Analysis", "⚖️ Bias & Credibility", "🔗 Similarity Analysis"])
        
        with tab1:
            st.markdown("### Sentiment Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Transformer-Based Analysis**")
                st.write(f"Overall Sentiment: **{sentiment_data['transformer_label']}**")
                st.write(f"Confidence: **{sentiment_data['transformer_confidence']:.2%}**")
                st.markdown("---")
                display_sentiment_chart(sentiment_data)
            
            with col2:
                st.markdown("**TextBlob Analysis**")
                st.write(f"Polarity: **{sentiment_data['polarity']:.3f}** (-1 to +1)")
                st.write(f"Subjectivity: **{sentiment_data['subjectivity']:.3f}** (0 to 1)")
                st.markdown("---")
                st.markdown("**Interpretation:**")
                if sentiment_data['polarity'] > 0.1:
                    st.write("✅ Positive tone detected")
                elif sentiment_data['polarity'] < -0.1:
                    st.write("⚠️ Negative tone detected")
                else:
                    st.write("➖ Neutral tone")
        
        with tab2:
            st.markdown("### Bias Indicators")
            col1, col2 = st.columns([2, 1])
            with col1:
                display_bias_breakdown(bias_metrics, sentiment_data)
            
            with col2:
                st.markdown("**Credibility Score**")
                st.markdown(f"# {credibility:.0f}/100")
                if credibility >= 70:
                    st.success("✅ High credibility - appears factual and balanced")
                elif credibility >= 50:
                    st.warning("⚠️ Moderate credibility - exercise critical thinking")
                else:
                    st.error("❌ Low credibility - highly subjective tone")

        with tab3:
            display_similarity_results(similarity_results)

if __name__ == "__main__":
    main()