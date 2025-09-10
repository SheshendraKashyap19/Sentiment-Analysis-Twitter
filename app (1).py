#import streamlit as st
import pickle
#import spacy
import numpy as np
import spacy
import streamlit as st

st.write("spaCy version:", spacy.__version__)


# Load spaCy embeddings
nlp = spacy.load("en_core_web_md")


# Load trained model
with open('logistic_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load spaCy embeddings
nlp = spacy.load('en_core_web_md')

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

st.title("Tweet Sentiment Analyzer")
st.write("Enter a tweet below to predict its sentiment:")

tweet_input = st.text_area("Your Tweet:")
if st.button("Predict Sentiment"):
    if tweet_input.strip() != "":
        result = predict_sentiment(tweet_input)
        st.success(f"Predicted Sentiment: {result}")
    else:
        st.error("Please enter a tweet!")
