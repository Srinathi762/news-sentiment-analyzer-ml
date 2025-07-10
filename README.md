
# 📰 News Sentiment Analyzer using Machine Learning

This repository contains a machine learning project that performs sentiment analysis on news text, classifying it into **Positive**, **Negative**, or **Neutral** categories. The project uses **Natural Language Processing (NLP)** and **Logistic Regression** to process and model real-world news data.

---

## 📌 Project Objective

To develop a sentiment analysis model that:
- Analyzes raw news headlines and descriptions
- Detects the underlying sentiment (positive, neutral, negative)
- Provides real-time predictions for user-input sentences
- Uses a clean, explainable machine learning pipeline

---

## 🗃️ Dataset

- **Source**: [Kaggle – https://www.kaggle.com/datasets/clovisdalmolinvieira/news-sentiment-analysis]
- **Fields Used**:
  - `Description` – News content to analyze
  - `Sentiment` – Annotated sentiment label

---

## 🧠 Model Workflow

1. **Data Preprocessing**
   - Remove special characters, links, punctuation
   - Lowercasing and stripping whitespaces
2. **Label Encoding** (`Positive`, `Neutral`, `Negative`)
3. **Feature Extraction** with TF-IDF
4. **Model Training** using Logistic Regression with `class_weight='balanced'`
5. **Model Evaluation**
   - Confusion Matrix
   - Classification Report
   - Visualizations (heatmap, learning curve, ROC curve)
6. **User Input Sentiment Prediction**
   - Integrated in Google Colab with form-based input

---

## 🔧 Tools & Technologies

- Python 3.9+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab

---

## 🧪 Sample Predictions

| Input Sentence                                      | Predicted Sentiment |
|----------------------------------------------------|----------------------|
| "The product exceeded my expectations."            | Positive 😊           |
| "The presentation is scheduled for tomorrow."      | Neutral 😐            |
| "I'm highly dissatisfied with the experience."     | Negative 😡           |

---

## 💾 Model & Vectorizer Files
  
To reuse the trained model:
```python
import joblib

# Save model
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Load later
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')



