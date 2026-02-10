import streamlit as st
import joblib
import json

# Load models & vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")
lr = joblib.load("imdb_logistic_model.pkl")
nb = joblib.load("naive_bayes_imdb.pkl")
svm = joblib.load("svm_imdb.pkl")
rf = joblib.load("rf_imdb.pkl")

# Load accuracies (proper logic)
with open("model_accuracies.json", "r") as f:
    accuracies = json.load(f)

st.title("🎬 IMDb Movie Review Sentiment Analyzer")

review = st.text_area("Enter your movie review:")

model_choice = st.selectbox(
    "Select Model",
    ("Logistic Regression", "Naive Bayes", "SVM", "Random Forest")
)

if st.button("Predict Sentiment"):
    review_vec = tfidf.transform([review])

    if model_choice == "Logistic Regression":
        pred = lr.predict(review_vec)[0]
    elif model_choice == "Naive Bayes":
        pred = nb.predict(review_vec)[0]
    elif model_choice == "SVM":
        pred = svm.predict(review_vec)[0]
    else:
        pred = rf.predict(review_vec)[0]

    # ✅ COLOR BASED ON MODEL OUTPUT (0 or 1) — NOT WORDS
    if pred == 1:
        st.markdown(
            "<div style='background-color: green; padding: 10px; border-radius: 5px;'>"
            "<h4>✅ Predicted Sentiment: Positive</h4></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color: red; padding: 10px; border-radius: 5px;'>"
            "<h4>❌ Predicted Sentiment: Negative</h4></div>",
            unsafe_allow_html=True
        )

    # Show only selected model accuracy
    st.subheader("📊 Model Performance (From Training)")
    st.write(f"{model_choice} Accuracy: {accuracies[model_choice]:.2f}")
  
