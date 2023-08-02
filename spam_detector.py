import pickle
import numpy as np
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Function to process the text
def process_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()                                               # Convert to lower case
    return text

# Load the trained model and the vectorizer
clf = pickle.load(open('finalized_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def classify_email(text):
    # Preprocess the text
    text = process_text(text)
    # Transform the text to match the input of the model
    text_vectorized = vectorizer.transform([text])
    # Predict the class of the text
    prediction = clf.predict(text_vectorized)
    # Return the prediction
    if prediction[0] == 1:
        return "This email is SPAM"
    else:
        return "This email is NOT SPAM"

# Load CSV file
data = pd.read_csv('your_file.csv')

# Apply classification to each row of the dataset
data['classification'] = data['text'].apply(classify_email)

# Print classifications
print(data['classification'])