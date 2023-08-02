# Naive Bayes Spam Classifier

This project uses Natural Language Processing (NLP) to train a Naive Bayes classifier to categorize emails as either 'spam' or 'not spam'. 

## Context

This project serves as an example of how NLP and machine learning can be used to automatically detect unwanted emails in a large dataset. 

## Content

The dataset used in this project is `mails.csv`, which is available on Kaggle at this [link](https://www.kaggle.com/datasets/karthickveerakumar/spam-filter). It consists of labeled emails classified as either 'spam' or 'not spam'.

## Acknowledgements

The dataset used in this project was sourced from Kaggle, and we would like to thank the provider Karthick Veerakumar for making this dataset available to the public.

## Inspiration

This project provides a practical example of how machine learning can be used to filter out spam, improving the quality of digital communication. We hope this project inspires further research into spam detection and other NLP applications.

## Requirements

- Python 3.7 or above
- pandas
- numpy
- scikit-learn
- pickle

## Instructions

1. Clone the repository.
2. Install the required libraries.
3. Download the `mails.csv` file from Kaggle and place it in the same directory as the cloned repository.
4. Run `spam_classifier.py` to train the model and save it. This will generate `finalized_model.pkl` (the trained model) and `vectorizer.pkl` (the trained vectorizer).
5. Run `spam_detector.py` to classify new emails. Replace `"Your input text here"` with the email you want to classify.