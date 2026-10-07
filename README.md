# 🛡️ Veritas AI — Misinformation Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)


Veritas AI is an end-to-end Machine Learning pipeline and interactive Streamlit web application designed to identify fake news articles in real time. Combining **TF-IDF vectorization**, **sentiment/emotion analysis**, and **regularized Lasso ($L_1$) Logistic Regression**, the system achieves high classification accuracy while maintaining lightweight, CPU-efficient inferencing.

---

## 🚀 Key Features

* **Hybrid Text & Metadata Vectorization:** Combines n-gram TF-IDF representations of `fully_combined_article` with structural text features (`char_count`, `word_count`, `avg_word_len`).
* **Zero-Bias Categorical Encoding:** One-hot encodes nominal NLP signals (`predicted_sentiment` & `predicted_emotion`) without introducing rank bias.
* **Regularized Feature Selection:** $L_1$ Lasso regularization eliminates uninformative text tokens, reducing overfitting and optimizing inference speed.
* **Modern Dark-Mode UI:** Glassmorphism dashboard with dynamic probability indicators and detailed verification metrics.

---

## 📊 Model Performance

Evaluated on an 80/20 train/test split with Stratified 5-Fold Cross-Validation:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.97%** |
| **Recall (Fake News Catch Rate)** | **99.33%** |
| **Precision** | **98.67%** |
| **F1-Score** | **0.9900** |
| **ROC-AUC** | **0.9982** |

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit web application
├── fake_news_detection_model   # Trained Lasso Logistic Regression model
├── tfidf_vectorizer            # Fitted TF-IDF vectorizer (max_features=10000)
├── one_hot_encoder             # Fitted OneHotEncoder for sentiment & emotion
├── scaler.pkl                  # Fitted StandardScaler for numerical metadata
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
