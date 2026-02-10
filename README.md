# IMDb Movie Review Sentiment Analysis Report

## Introduction
This report presents a sentiment analysis system that automatically classifies IMDb movie reviews as either Positive or Negative. The project applies supervised machine learning
algorithms to analyze text data and predict sentiment. The goal is to build an accurate, reusable and user-friendly model that can be deployed as a web application.

## Problem Statement 
Movie reviews contain rich opinions in unstructured text form. Manually reading and classifying thousands of reviews is time-consuming and impractical. Therefore, the problem is to
develop an automated system that can correctly determine whether a given movie review expresses a positive or negative sentiment.

## Dataset Description
The dataset used in this project is the IMDb Movie Review Dataset, obtained from Kaggle. It contains 50,000 movie reviews in total, divided equally into 25,000 training samples and 25,000 testing samples. Each review is labeled as either 'positive' or 'negative', making this a binary classification problem. The dataset is widely used in NLP research and sentiment analysis tasks.

## Methodology
The methodology followed a structured machine learning pipeline. First, exploratory data analysis (EDA) was performed to understand the dataset, check for missing values, and analyze review length distribution. Then, text preprocessing was applied by converting sentiment labels into numerical format (Positive = 1, Negative = 0). The text data was transformed into numerical feature vectors using TF-IDF Vectorization with English stop words removed and a maximum of 5,000 features. The dataset was split into training and testing sets. Four machine learning models were trained: Logistic Regression, Multinomial Naive Bayes, Support Vector Machine (SVM), and Random Forest Classifier.

## Model Evaluation and Results
Each model was evaluated using standard classification metrics including Accuracy, Precision, Recall, F1-score, and Confusion Matrix. The results were compared in a summary table to 
determine the best-performing model. Logistic Regression and SVM generally performed strongly for high-dimensional text data, while Naive Bayes provided fast and efficient baseline 
performance. Random Forest offered an ensemble-based approach for comparison.

## Discussion
The experiment demonstrated that linear models such as Logistic Regression and SVM are highly effective for text-based sentiment analysis when combined with TF-IDF features. Naive Bayes, while simpler, performed competitively and is useful for lightweight applications. Random Forest was more computationally expensive but helped in understanding different model behaviors. Overall, model performance depended on feature representation and algorithm selection.

## Conclusion
This project successfully implemented an end-to-end sentiment analysis system for IMDb movie reviews, including data preprocessing, feature extraction, model training, evaluation and 
deployment. All trained models were saved using Joblib and integrated into a Streamlit web application, allowing users to input a review and select a model for prediction. In the future, deep learning models such as LSTM, BERT, or Transformer-based architectures can be explored to further improve accuracy and performance.
