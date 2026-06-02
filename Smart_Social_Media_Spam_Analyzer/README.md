# 🤖 AI Spam Analyzer

## 📌 Project Overview

AI Spam Analyzer is a Machine Learning and NLP-based web application that detects whether a social media comment is Spam or Genuine. The application uses TF-IDF feature extraction and Logistic Regression for classification, while also providing sentiment analysis, spam explanations, word cloud visualization, and model comparison.

---

## 🚀 Features

### ✅ Spam Detection

* Detects spam and genuine comments.
* Provides prediction confidence score.

### 😊 Sentiment Analysis

* Analyzes comment sentiment:

  * Positive
  * Negative
  * Neutral

### 🧠 Spam Explanation

* Identifies suspicious patterns:

  * Promotional keywords
  * Suspicious links
  * Excessive capital letters
  * Very short comments

### 📊 Live Dashboard

* Total comments
* Spam comments
* Genuine comments
* Spam percentage
* Genuine percentage

### 📈 Prediction Visualization

* Bar chart showing spam and genuine probabilities.
* Confidence indicator with progress bar.

### ☁️ Word Cloud

* Displays common words used in spam comments.

### 🤖 Model Comparison

Compares multiple machine learning models:

* Logistic Regression
* Naive Bayes
* Random Forest

### 📂 CSV Upload

* Upload CSV files for viewing and analysis.

### 📥 Report Download

* Download prediction reports in CSV format.

---

## 🛠 Technologies Used

* Python
* Streamlit
* Pandas
* NLTK
* Scikit-Learn
* Matplotlib
* TextBlob
* WordCloud

---

## 🧠 Machine Learning Workflow

1. Load Dataset
2. Clean Text Data
3. Remove Stopwords
4. TF-IDF Vectorization
5. Train Machine Learning Models
6. Predict Spam/Genuine Comments
7. Display Results and Analytics

---

## 📂 Project Structure

AI-Spam-Analyzer/

├── app.py

├── spam.csv

├── requirements.txt

├── README.md

└── screenshots/

    ├── dashboard.png

    ├── prediction.png

    └── wordcloud.png

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AI-Spam-Analyzer.git
cd AI-Spam-Analyzer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📊 Sample Output

* Spam Detection Result
* Sentiment Analysis
* Confidence Score
* Prediction Graph
* Word Cloud
* Model Accuracy Comparison

---

## 📈 Future Improvements

* Deep Learning Models (LSTM/BERT)
* Real-Time Social Media Integration
* Multi-Language Spam Detection
* User Authentication
* Cloud Deployment

---

## 👩‍💻 Author

Akarapu Srivalli

Mini Project – Machine Learning & NLP

---

## ⭐ Project Outcome

This project demonstrates practical implementation of Natural Language Processing and Machine Learning techniques for detecting spam comments and analyzing social media content through an interactive Streamlit dashboard.
