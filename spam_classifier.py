import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import string

# Load the data
data = pd.read_csv("mails.csv")

# Function to process the text
def process_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()                                               # Convert to lower case
    return text

# Apply the function to the 'text' column
data['text'] = data['text'].apply(process_text)

# Initialize the CountVectorizer
vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the 'text' column
word_count = vectorizer.fit_transform(data['text'])

# Separate spam and ham
spam_indices = data[data['spam']==1].index
ham_indices = data[data['spam']==0].index

spam_word_count = word_count[spam_indices]
ham_word_count = word_count[ham_indices]

# Calculate word frequencies
spam_word_freq = np.sum(spam_word_count.toarray(), axis=0)
ham_word_freq = np.sum(ham_word_count.toarray(), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(word_count, data['spam'], test_size=0.3, random_state=0)

# Instantiate the Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Print the classification report
print(classification_report(y_test, y_pred))

# Save the model and vectorizer to disk
pickle.dump(clf, open('finalized_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))


print(data.head())