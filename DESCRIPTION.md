# Product Review Sentiment Analyzer

A machine learning project that predicts whether an Amazon product review is Positive or Negative using Natural Language Processing (NLP) techniques.

---

## What This Project Does

You type a product review in the terminal, and the model instantly tells you:
- Whether the review is POSITIVE or NEGATIVE
- How confident the model is (confidence score in %)

---

## Example

```
Your Review: This product is absolutely amazing, works perfectly!

  Sentiment  :  POSITIVE :)
  Confidence :  94.21%

Your Review: Waste of money, broke after one day.

  Sentiment  :  NEGATIVE :(
  Confidence :  88.45%
```

---

## How It Works

1. Loads Amazon product review dataset (5000 reviews)
2. Labels reviews — ratings 4 and 5 = Positive, ratings 1 to 3 = Negative
3. Fixes class imbalance using oversampling so the model learns both classes equally
4. Cleans text by removing punctuation, numbers, and stopwords
5. Converts cleaned text into numbers using TF-IDF Vectorization
6. Trains a Logistic Regression classifier on the data
7. Takes live user input from terminal and predicts sentiment in real time

---

## Tech Stack

- Python
- Pandas — data handling
- NLTK — stopword removal
- Scikit-learn — TF-IDF, Logistic Regression, model evaluation

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~91% |
| Positive Precision | High |
| Negative Recall | Improved via class balancing |

---

## Dataset

- Source: Datafiniti Amazon Consumer Reviews dataset (available on Kaggle)
- Size: 5000 reviews
- Features used: Review text and star rating

---

## How To Run

```bash
pip install -r requirements.txt
python sentiment.py
```

Then type any product review in the terminal and press Enter to get an instant prediction.

---

## Files

| File | Description |
|------|-------------|
| sentiment.py | Main program with model training and user input loop |
| requirements.txt | All Python libraries needed |
| README.md | Project documentation |
| DESCRIPTION.md | This file — project overview |
