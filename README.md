# 📰 News Article Bias & Similarity Analyzer

A production-ready web application that analyzes news articles for sentiment, bias, credibility, and semantic similarity using state-of-the-art NLP models.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

This application helps students, researchers, and journalists evaluate news quality by:
- **Extracting** clean article content from any news URL
- **Analyzing** sentiment using transformer models (DistilBERT)
- **Detecting** bias through subjectivity and emotional language analysis
- **Scoring** credibility based on objectivity metrics
- **Finding** semantically similar articles using sentence embeddings

## ✨ Features

### Core Analysis
- ✅ **URL-based article extraction** - Supports most major news sites
- ✅ **Dual sentiment analysis** - TextBlob + Transformer models
- ✅ **Bias detection** - Subjectivity, emotional language, intensity metrics
- ✅ **Credibility scoring** - 0-100 scale based on objectivity
- ✅ **Semantic similarity** - Compare with reference articles using embeddings

### User Experience
- 📊 **Interactive visualizations** - Charts, metrics cards, progress bars
- 💾 **CSV export** - Download analysis results
- 🎨 **Clean UI** - Professional Streamlit interface
- ⚡ **Model caching** - Fast subsequent analyses

## 🛠️ Tech Stack

**Backend/ML:**
- Python 3.8+
- Transformers (DistilBERT for sentiment)
- Sentence-Transformers (MiniLM for embeddings)
- Newspaper3k (article extraction)
- TextBlob (linguistic analysis)
- NLTK (text processing)
- Scikit-learn (similarity computation)

**Frontend:**
- Streamlit (web framework)

## 📁 Project Structure

```
news-analyzer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .streamlit/           # Streamlit config (optional)
│   └── config.toml
└── data/                 # Cache directory (auto-created)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM (for transformer models)
- Internet connection (for article extraction)

### Installation

1. **Clone or download this project**
```bash
cd news-analyzer
```

2. **Create a virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

This will install all required packages. First-time installation may take 5-10 minutes due to large ML models.

### Running the Application

1. **Start the Streamlit server**
```bash
streamlit run app.py
```

2. **Open your browser**
The app will automatically open at `http://localhost:8501`

If it doesn't open automatically, manually navigate to the URL shown in the terminal.

3. **Analyze an article**
- Paste a news article URL (e.g., from BBC, Reuters, Guardian)
- Click "Analyze Article"
- Wait 10-30 seconds for processing
- View comprehensive analysis results

## 📖 Usage Guide

### Example URLs to Try
```
https://www.bbc.com/news/science-environment-58874502
https://www.reuters.com/technology/
https://www.theguardian.com/environment/climate-crisis
```

### Understanding the Results

**Sentiment Analysis**
- Shows positive/negative tone using transformer models
- Polarity: -1 (very negative) to +1 (very positive)
- Confidence score indicates model certainty

**Bias & Credibility**
- **Subjectivity**: 0% = pure facts, 100% = pure opinion
- **Emotional Bias**: Strength of emotional language
- **Language Intensity**: Frequency of strong modifiers
- **Credibility Score**: Composite 0-100 rating (higher = more credible)

**Similarity Analysis**
- Compares article to reference corpus
- Uses semantic embeddings (meaning-based, not keyword matching)
- Scores show topical and framing similarity

### Exporting Results
Click "Download Analysis (CSV)" to save results for further analysis or reporting.

## 🔧 Configuration

### Model Selection
The app uses these pre-trained models (cached on first run):

```python
# Sentiment Analysis
"distilbert-base-uncased-finetuned-sst-2-english"

# Semantic Similarity
"all-MiniLM-L6-v2"
```

To use different models, modify the `load_sentiment_model()` and `load_sentence_transformer()` functions in `app.py`.

### Reference Articles
Reference articles for similarity comparison are defined in `get_reference_articles()`. 

**To customize:**
1. Open `app.py`
2. Find the `get_reference_articles()` function
3. Replace with your own articles or connect to a database/API

### Performance Tuning
- **Memory**: Models require ~2GB RAM. Reduce by using smaller models
- **Speed**: First run downloads models (~500MB). Subsequent runs are fast
- **Accuracy**: Larger models (BERT-base) provide better results but slower processing

## 🧪 Development

### Code Structure

**Main Components:**
- `extract_article()` - Article extraction pipeline
- `analyze_sentiment()` - Sentiment analysis using dual models
- `calculate_bias_metrics()` - Bias detection algorithms
- `calculate_credibility_score()` - Credibility computation
- `compute_similarity()` - Semantic similarity engine

**Caching Strategy:**
```python
@st.cache_resource  # Models (loaded once)
@st.cache_data      # Reference data (loaded once)
```

### Adding New Features

The code includes clearly marked extension points:

```python
# ===== FUTURE IMPROVEMENT =====
# - Political bias classification
# - Sentence-level fact vs opinion
# - Real-time news API integration
# - Multilingual support
```

## 🐛 Troubleshooting

### Common Issues

**"Failed to extract article"**
- Site may block scraping (paywall/bot protection)
- Try mainstream news sources (BBC, Reuters, etc.)
- Check URL is accessible in browser

**"Models taking too long to load"**
- First run downloads ~500MB of models
- Subsequent runs use cache (much faster)
- Check internet connection

**"Out of memory"**
- Transformer models need 2-4GB RAM
- Close other applications
- Use smaller batch sizes (reduce `chunks` in sentiment analysis)

**"Import errors"**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+)

### Debug Mode
```bash
streamlit run app.py --logger.level=debug
```

## 📊 Model Information

### Sentiment Analysis Model
- **Model**: DistilBERT SST-2
- **Training**: Stanford Sentiment Treebank
- **Accuracy**: ~92% on benchmark datasets
- **Size**: ~250MB

### Embedding Model
- **Model**: all-MiniLM-L6-v2
- **Purpose**: Sentence embeddings for similarity
- **Performance**: 384-dimensional embeddings
- **Size**: ~80MB

## 🎓 Educational Use

This project demonstrates:
- Production-ready ML application architecture
- Transformer model integration (Hugging Face)
- Streamlit web framework
- NLP pipeline design
- Model caching and performance optimization
- Clean code organization

Perfect for:
- Academic portfolios
- Job applications (data science, ML engineering)
- Research projects
- Teaching NLP concepts

## 🔮 Future Enhancements

Planned improvements (commented in code):

1. **Political Bias Classification**
   - Left/center/right spectrum analysis
   - Pre-trained political bias models

2. **Sentence-Level Analysis**
   - Fact vs opinion highlighting
   - Quote detection and attribution
   - Named entity analysis

3. **Advanced Similarity**
   - Real-time news API integration
   - Topic clustering
   - Temporal analysis

4. **Multilingual Support**
   - Translation pipeline
   - Language-specific models

5. **Chrome Extension**
   - One-click article analysis
   - In-browser results overlay

6. **Fake News Detection**
   - Claim verification
   - Source credibility database

## 📄 License

MIT License - Feel free to use for academic or commercial projects.

## 🤝 Contributing

This is a demonstration project. For production use:
- Add comprehensive error handling
- Implement API rate limiting
- Set up proper logging
- Add unit tests
- Deploy to cloud platform (Streamlit Cloud, Heroku, AWS)

## 📧 Contact

For questions or feedback about this project:
- Open an issue on GitHub
- Use for educational/portfolio purposes
- Customize freely for your needs

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [Hugging Face Transformers](https://huggingface.co/) - NLP models
- [Newspaper3k](https://newspaper.readthedocs.io/) - Article extraction
- [Sentence-Transformers](https://www.sbert.net/) - Semantic embeddings

---




