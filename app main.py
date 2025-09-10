import pickle
import streamlit as st
import numpy as np

# Load vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load trained model
with open("logistic_sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
