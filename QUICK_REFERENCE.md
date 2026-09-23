# Quick Reference Card

## 🚀 30-Second Start Guide

```bash
# 1. Setup (first time only)
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Run
streamlit run app.py

# 3. Use
# Paste article URL → Click "Analyze Article" → View results
```

---

## 📊 Understanding Your Results

### Credibility Score
| Score | Meaning | What to Do |
|-------|---------|------------|
| 80-100 | Excellent | Trust the facts presented |
| 70-79 | Good | Generally reliable |
| 50-69 | Moderate | Cross-check important claims |
| 30-49 | Low | Verify with other sources |
| 0-29 | Very Low | High bias - be skeptical |

### Bias Category
- **Low Bias (Fact-Based)**: ✅ Objective reporting
- **Moderate Bias**: ⚠️ Some opinion mixed in
- **High Bias (Opinion-Heavy)**: ⚠️ Strong editorial slant

### Subjectivity
- **0-30%**: Mostly facts
- **30-60%**: Balanced mix
- **60-100%**: Mostly opinions

---

## 🎯 Common Commands

```bash
# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

# Run tests
python test_setup.py

# Deactivate virtual environment
deactivate
```

---

## 🔧 Troubleshooting Quick Fixes

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Failed to extract article"
- ✅ Try BBC, Reuters, Guardian URLs
- ❌ Avoid paywalled sites (NYT, WSJ)

### "Models downloading"
- ⏱️ First run takes 5-10 minutes
- 📦 Downloads ~500MB
- ✅ Subsequent runs are fast

### "Out of memory"
- Close other applications
- Restart computer
- Need 4GB+ RAM

---

## 📁 File Locations

```
news-analyzer/
├── app.py              ← Main application
├── requirements.txt    ← Dependencies
├── README.md          ← Full documentation
├── setup.sh/.bat      ← Auto-setup scripts
└── test_setup.py      ← Test everything works
```

---

## 🌐 Example URLs to Try

```
https://www.bbc.com/news
https://www.reuters.com/world
https://www.theguardian.com/international
```

Avoid:
- Paywalled sites (Wall Street Journal, New York Times)
- Social media links
- PDF documents

---

## 💡 Quick Tips

1. **First Analysis**: Takes 10-30 seconds (downloads models)
2. **Subsequent**: Much faster (models cached)
3. **Best Results**: Mainstream news sites
4. **Export Data**: Click "Download Analysis (CSV)"
5. **Compare**: Analyze multiple articles on same topic

---

## 📞 Getting Help

1. **Check README.md** - Comprehensive guide
2. **Run test_setup.py** - Diagnose issues
3. **Read EXAMPLES.md** - See use cases
4. **Check DEPLOYMENT.md** - Cloud hosting

---

## 🎓 For Presentations/Demos

**Talking Points**:
1. "Analyzes news articles for bias and credibility"
2. "Uses state-of-the-art NLP models (transformers)"
3. "Provides objective metrics: sentiment, subjectivity, credibility"
4. "Built with Python, Streamlit, and Hugging Face models"
5. "Production-ready code with proper architecture"

**Live Demo**:
1. Open app: `streamlit run app.py`
2. Paste BBC news URL
3. Click analyze
4. Explain metrics while processing
5. Show results in ~20 seconds
6. Export to CSV

**Key Features to Highlight**:
- ✅ Clean, professional UI
- ✅ Real NLP models (not fake/demo)
- ✅ Multiple analysis dimensions
- ✅ Export functionality
- ✅ Well-documented code
- ✅ Production architecture

---

## 🏆 Project Highlights

**For Recruiters**:
- Production-ready code structure
- ML model integration (Hugging Face)
- Clean separation of concerns
- Comprehensive documentation
- Error handling throughout
- Performance optimization (caching)
- Modern Python practices

**Technologies Demonstrated**:
- Python 3.8+
- Streamlit (web framework)
- Transformers (NLP)
- Sentence-Transformers (embeddings)
- Pandas/NumPy (data processing)
- NLTK (text processing)

**ML Engineering Skills**:
- Model loading & caching
- Inference optimization
- Batch processing
- Memory management
- API integration (Hugging Face)
- Production deployment patterns

---

## ⚡ Performance Notes

| Metric | Value |
|--------|-------|
| First Run | 30-60 seconds |
| Model Loading | 5-10 seconds |
| Per Article | 10-30 seconds |
| RAM Required | 2-4 GB |
| Disk Space | 2 GB |

---

## 📈 Metrics Explained (1-Sentence Each)

- **Sentiment**: Positive, negative, or neutral tone
- **Polarity**: -1 (very negative) to +1 (very positive)
- **Subjectivity**: 0% (pure facts) to 100% (pure opinion)
- **Emotional Bias**: How much emotional language is used
- **Credibility**: 0-100 score based on objectivity
- **Similarity**: How similar to reference articles (0-100%)

---

## 🔮 Extending the Project

**Easy Additions** (1-2 hours):
- Add more reference articles
- Customize UI theme
- Add new metrics

**Medium Additions** (1-2 days):
- Database integration
- REST API wrapper
- User authentication

**Complex Additions** (1-2 weeks):
- Political bias classifier
- Real-time news API
- Multi-language support
- Chrome extension

See `API.md` and code comments for details.

---

## 📋 Pre-Demo Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Test run completed successfully
- [ ] Example URLs prepared
- [ ] Internet connection verified
- [ ] Browser ready (for localhost:8501)
- [ ] Talking points reviewed
- [ ] CSV export tested

---

## 🎤 Elevator Pitch

*"I built a news article analyzer that uses transformer models to detect bias, measure credibility, and compare articles. It's a full-stack ML application with a clean web interface, production-ready code, and comprehensive documentation. Perfect for students, journalists, or anyone evaluating news quality."*

---

**Print this page for quick reference during demos and presentations!**
