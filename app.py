
import streamlit as st
import pickle
import numpy as np

# Load TF-IDF vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('logistic_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

positive_keywords = ['love', 'amazing', 'awesome', 'fantastic', 'best', 'happy', 'great', 'excellent']
negative_keywords = ['hate', 'terrible', 'awful', 'worst', 'disappointing', 'sad']

def predict_sentiment(tweet):
    tweet_lower = tweet.lower()
    if any(word in tweet_lower for word in positive_keywords):
        return "Positive"
    elif any(word in tweet_lower for word in negative_keywords):
        return "Negative"
    
    vector = vectorizer.transform([tweet])
    pred = model.predict(vector)[0]
    return "Positive" if pred == 1 else "Negative"

st.title("Tweet Sentiment Analyzer")
st.write("Enter a tweet below to predict its sentiment:")

tweet_input = st.text_area("Your Tweet:")
if st.button("Predict Sentiment"):
    if tweet_input.strip() != "":
        result = predict_sentiment(tweet_input)
        st.success(f"Predicted Sentiment: {result}")
    else:
        st.error("Please enter a tweet!")
