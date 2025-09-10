import pickle
import numpy as np
import spacy
import streamlit as st

# Show spaCy version (for debugging)
st.write("spaCy version:", spacy.__version__)

# Cache the model so it doesn’t reload on every rerun
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_md")

nlp = load_model()

# Load trained ML model
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
    
    vector = nlp(tweet).vector.reshape(1, -1)
    pred = model.predict(vector)[0]
    return "Positive" if pred == 1 else "Negative"

# Streamlit UI
st.title("Tweet Sentiment Analyzer")
st.write("Enter a tweet below to predict its sentiment:")

tweet_input = st.text_area("Your Tweet:")
if st.button("Predict Sentiment"):
    if tweet_input.strip() != "":
        result = predict_sentiment(tweet_input)
        st.success(f"Predicted Sentiment: {result}")
    else:
        st.error("Please enter a tweet!")
