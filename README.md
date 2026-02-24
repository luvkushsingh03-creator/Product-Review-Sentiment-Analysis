# Product Review Sentiment Analyzer

A machine learning model that analyzes the sentiment of Amazon product reviews using **Logistic Regression** and **TF-IDF Vectorization.

# Features
- Classifies reviews as **Positive** or **Negative**
- Shows **confidence score** for each prediction
- Handles class imbalance using oversampling
- Interactive console-based user input

# Dataset
Uses the **Datafiniti Amazon Consumer Reviews** dataset.
- Download from Kaggle: [Amazon Product Reviews](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
- Place the CSV file in the same folder as `sentiment.py`
- Rename it to: `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`



      MODEL READY!
    Accuracy : 91.5%


----------------------------------------
Your Review: This product is absolutely amazing!

  Sentiment  :  POSITIVE :)
  Confidence :  94.21%

----------------------------------------
Your Review: Worst product ever, broke in one day.

  Sentiment  :  NEGATIVE :(
  Confidence :  88.45%

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading and processing |
| NLTK | Stopword removal |
| Scikit-learn | TF-IDF, Logistic Regression, Train/Test split |

## Project Structure
sentiment-analyzer/
│
├── sentiment.py        # Main program
├── requirements.txt    # Dependencies
└── README.md           # Project documentation

## How It Works
1. Loads Amazon product review dataset
2. Converts ratings (4-5 = Positive, 1-3 = Negative)
3. Fixes class imbalance using oversampling
4. Cleans text (lowercase, remove punctuation, remove stopwords)
5. Converts text to numbers using TF-IDF
6. Trains a Logistic Regression model
7. Takes user input and predicts sentiment in real time

#Dataset
Download form this link : https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products

# Author
Made by Luvkush Singh
