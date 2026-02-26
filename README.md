# 🎬 IMDb Movie Review Sentiment Analysis

- A Machine Learning based web application that classifies IMDb movie reviews as Positive or Negative.

--- 

# 🚀 Project Overview

- This project uses Natural Language Processing (NLP) and supervised Machine Learning techniques to analyze and classify movie reviews based on sentiment.

- The model is trained on the Kaggle IMDb Movie Review Dataset and deployed using Streamlit for real-time predictions.

---
# 🛠 Technologies Used

- Python

- Pandas

- NumPy

- NLTK

- Scikit-learn

- TF-IDF Vectorizer

- Logistic Regression

- Multinomial Naive Bayes

- Support Vector Machine (SVM)

- Random Forest

- Joblib

- Streamlit
---
# 📊 Dataset Details

- Dataset: IMDb Movie Review Dataset

- Source: Kaggle

- Total Reviews: 50,000

- Training Data: 25,000

- Testing Data: 25,000

Labels:

- Positive = 1

- Negative = 0

Type: Binary Classification Problem

---

# 🔍 Methodology
## 1. Text Preprocessing

- Converting text to lowercase

- Removing stopwords

- Tokenization

- Stemming

- Converting labels into numerical format

## 2. Feature Extraction

- Text transformed using TF-IDF Vectorization

- Maximum 5,000 features

- English stopwords removed

## 3. Model Training

Multiple machine learning models were trained and compared:

- Logistic Regression

- Multinomial Naive Bayes

- Support Vector Machine (SVM)

- Random Forest Classifier

---

# 📈 Model Evaluation

Evaluation Metrics Used:

- Accuracy

- Precision

- Recall

- F1-Score

- Confusion Matrix

---

# 🏆 Best Performing Models

- Logistic Regression

- Support Vector Machine (SVM)

Linear models performed exceptionally well for high-dimensional text data.

---

# 💡 Key Insights

- TF-IDF works very effectively for sentiment classification.

- Linear models like Logistic Regression and SVM perform strongly on text data.

- Naive Bayes provides fast and efficient baseline performance.

- Random Forest is computationally heavier but useful for comparison.

---

# 🌐 Deployment

- All trained models were saved using Joblib.

- Integrated into a Streamlit web application.

Users can:

- Enter a movie review

- Select a model

- Get instant sentiment prediction

---

# 🔮 Future Improvements

Implement Deep Learning models like:

- LSTM

- BERT

- Transformer-based architectures

- Improve accuracy using advanced embeddings

- Add model comparison dashboard in the web app
