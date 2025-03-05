import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load NLTK stopwords
nltk.download('stopwords')

# Load trained model and vectorizer
svm_model = joblib.load('C:\\Users\\HP\\Desktop\\Project\\CODE TEC INTERNSHIP\\NLP task\\svm_model.pkl')
tfidf = joblib.load('C:\\Users\\HP\\Desktop\\Project\\CODE TEC INTERNSHIP\\NLP task\\tfidf_vectorizer.pkl')
# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters and numbers
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Streamlit App
st.title("Restaurant Review Sentiment Analysis")
st.write("Enter a review below to analyze sentiment:")

user_input = st.text_area("Review:")
if st.button("Analyze Sentiment"):
    if user_input:
        cleaned_input = preprocess_text(user_input)
        vectorized_input = tfidf.transform([cleaned_input])
        prediction = svm_model.predict(vectorized_input)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a review.")
