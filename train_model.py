import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Simple training dataset
data = {
    "text": [
        "Football match today",
        "Cricket world cup final",
        "Basketball tournament results",
        "New AI technology released",
        "Tech companies launch new smartphone",
        "Software development trends",
        "Healthy diet tips",
        "Exercise and fitness advice",
        "Medical research breakthrough",
        "Government election news",
        "Political debate in parliament",
        "Manga chapter released",
        "Anime episode review",
        "New comic series launched",
        "Chapter",
        "Manga",
        "Manhwa",
        "Product",
        "Price"
    ],
    "category": [
        "Sports", "Sports", "Sports",
        "Technology", "Technology", "Technology",
        "Health", "Health", "Health",
        "Politics", "Politics",
        "Entertainment", "Entertainment", "Entertainment", "Entertainment", "Entertainment", "Entertainment",
        "E-TECH", "E-TECH"
    ]
}

df = pd.DataFrame(data)

# UPGRADE 1: Use TF-IDF and remove common English words
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["text"])

model = MultinomialNB()
model.fit(X, df["category"])

# Save the upgraded model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Upgraded TF-IDF Model trained and saved!")