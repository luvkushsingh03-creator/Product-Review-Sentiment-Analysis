import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ======================================
# LOAD DATASET
# ======================================

print("\nLoading dataset...")
df = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
df.columns = df.columns.str.strip()
df = df[['reviews.text', 'reviews.rating']].dropna()
df['reviews.rating'] = pd.to_numeric(df['reviews.rating'], errors='coerce')
df.dropna(subset=['reviews.rating'], inplace=True)
df['sentiment'] = df['reviews.rating'].apply(lambda r: 1 if r >= 4 else 0)

print("Dataset Shape    :", df.shape)
print("Sentiment Counts :", df['sentiment'].value_counts().to_dict())

# ======================================
# FIX CLASS IMBALANCE
# ======================================

majority = df[df['sentiment'] == 1]
minority = df[df['sentiment'] == 0]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

df = pd.concat([majority, minority_upsampled])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Class balancing done.")

# ======================================
# TEXT CLEANING
# ======================================

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

print("Cleaning text...")
df['cleaned'] = df['reviews.text'].apply(clean_text)
df = df[df['cleaned'].str.strip() != '']

# ======================================
# TF-IDF + TRAIN MODEL
# ======================================

print("Training model...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

accuracy = round(model.score(X_test, y_test) * 100, 2)

print("\n" + "=" * 40)
print("      MODEL READY!")
print(f"      Accuracy : {accuracy}%")
print("=" * 40)

print("\nClassification Report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# ======================================
# USER INPUT LOOP
# ======================================

print("=" * 40)
print("   PRODUCT REVIEW SENTIMENT ANALYZER")
print("=" * 40)
print("Type your product review and press Enter.")
print("Type 'exit' to quit.\n")

while True:
    print("-" * 40)
    try:
        user_input = input("Your Review: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break

    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    if user_input == '':
        print("[!] Review cannot be empty. Try again.")
        continue

    cleaned = clean_text(user_input)

    if cleaned.strip() == '':
        print("[!] No meaningful words found. Try again.")
        continue

    vectorized  = vectorizer.transform([cleaned])
    prediction  = model.predict(vectorized)[0]
    proba       = model.predict_proba(vectorized)[0]
    confidence  = round(max(proba) * 100, 2)

    print()
    if prediction == 1:
        print("  Sentiment  :  POSITIVE :)")
    else:
        print("  Sentiment  :  NEGATIVE :(")
    print(f"  Confidence :  {confidence}%")
    print()
