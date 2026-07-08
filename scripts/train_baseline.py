"""
Step 3: Baseline Model - TF-IDF + Logistic Regression
--------------------------------------------------------
Trains a simple, fast baseline sentiment classifier. This gives you a
number to compare your DistilBERT + LoRA model against later, and it's
also a good "before vs after" story for your project write-up.
 
Run:
    python train_baseline.py
"""
 
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
INPUT_CSV = "../data/reviews_labeled.csv"
MODEL_OUTPUT = "../data/baseline_model.joblib"
VECTORIZER_OUTPUT = "../data/tfidf_vectorizer.joblib"
 
 
def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["review_text", "sentiment"])
    print(f"Loaded {len(df)} labeled reviews")
 
    X = df["review_text"]
    y = df["sentiment"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),   # unigrams + bigrams catch phrases like "not good"
        stop_words="english",
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
 
    print("Training Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # helps if neutral/negative classes are smaller
        random_state=42,
    )
    clf.fit(X_train_vec, y_train)
 
    y_pred = clf.predict(X_test_vec)
 
    acc = accuracy_score(y_test, y_pred)
    print(f"\nBaseline Accuracy: {acc * 100:.2f}%\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=clf.classes_))
    print("Labels order:", clf.classes_)
 
    joblib.dump(clf, MODEL_OUTPUT)
    joblib.dump(vectorizer, VECTORIZER_OUTPUT)
    print(f"\nSaved model to {MODEL_OUTPUT}")
    print(f"Saved vectorizer to {VECTORIZER_OUTPUT}")
 
 
if __name__ == "__main__":
    main()