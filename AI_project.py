# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:25:47 2024

@author: bmack
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout
from sklearn.preprocessing import LabelEncoder

def load_datasets(filenames):
    dfs = []
    for filename in filenames:
        file_location = filename
        df = pd.read_csv(file_location)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
filenames = ["cnbc_headlines.csv", "guardian_headlines.csv", "reuters_headlines.csv"]
# Load and preprocess datasets
df = load_datasets(filenames)
df = df.dropna().drop_duplicates()

# Vectorize the headlines using CountVectorizer with custom tokenizer
def tokenize_stem(text):
    blob = TextBlob(text)
    return [word.stem() for word in blob.words]
corpus = []
for item in df['Headlines']:
    corpus.append(' '.join(tokenize_stem(str(item))))
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 0].values
print(X.shape)
print(y.shape)  # Print first few rows of the vectorized features

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # Get polarity score (-1 to 1)
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to 'Headlines' or 'Description' to derive sentiment labels
df['Sentiment'] = df['Headlines'].apply(analyze_sentiment)

# Split the data into training and testing sets based on derived sentiment labels
X_train, X_test, y_train, y_test = train_test_split(df[['Headlines', 'Description']], df['Sentiment'], test_size=0.2, random_state=0)
# Vectorize and concatenate training data
cv = CountVectorizer()
X_train_headlines = cv.fit_transform(X_train['Headlines']).toarray()
X_train_description = cv.transform(X_train['Description']).toarray()
X_train_processed = np.concatenate((X_train_headlines, X_train_description), axis=1)

# Vectorize and concatenate testing data (using the same CountVectorizer)
X_test_headlines = cv.transform(X_test['Headlines']).toarray()
X_test_description = cv.transform(X_test['Description']).toarray()
X_test_processed = np.concatenate((X_test_headlines, X_test_description), axis=1)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_onehot = to_categorical(y_train_encoded)
y_test_encoded = label_encoder.transform(y_test)
y_test_onehot = to_categorical(y_test_encoded)

model = Sequential([
    Dense(128, input_dim=X_train_processed.shape[1], activation='relu'),
    Dropout(0.2),  
    Dense(64, activation='relu'),
    Dropout(0.3), 
    Dense(3, activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_processed, y_train_onehot, epochs=30, batch_size=32, verbose=1)

# Predict sentiment for test headlines using the trained neural network
y_pred = model.predict(X_test_processed)
y_pred = np.argmax(y_pred, axis=1)  # Convert softmax probabilities to class labels
print(y_pred)

sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return 2  # Positive sentiment
    elif compound_score <= 0.05:
        return 0  # Negative sentiment
    else:
        return 1  # Neutral sentiment

# Predict sentiment for test headlines using VADER
y_pred_vader = [analyze_sentiment(headline) for headline in df.loc[X_test.index]['Headlines']]

# Calculate accuracy
accuracy_nn = np.sum(y_pred == y_test_encoded) / len(y_test_encoded)
print("Neural Network Model Accuracy:", accuracy_nn)

# Convert VADER predictions to array
y_pred_vader = np.array(y_pred_vader)

# Calculate accuracy for VADER sentiment analysis
accuracy_vader = np.sum(y_pred_vader == y_test_encoded) / len(y_test_encoded)
print("VADER Sentiment Analysis Accuracy:", accuracy_vader)