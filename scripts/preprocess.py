"""
Step 2: Preprocessing & Sentiment Labeling (Kaggle IMDB 50K version)
------------------------------------------------------------------------
Adjusted for the Kaggle "IMDB Dataset of 50K Movie Reviews" CSV, which has
only two columns: `review` and `sentiment` (positive/negative - no star
rating, no film title, no neutral class).
 
Download from:
    https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Place the file as: ../data/IMDB Dataset.csv
 
Outputs:
1. reviews_labeled.csv      -> one row per review, with sentiment label
2. sentences_labeled.csv    -> one row per sentence (for BERTopic later)
 
Sentiment labeling logic (adjusted):
- The dataset's own `sentiment` column is the ground truth label (already
  positive/negative - no need to derive it from a star rating).
- VADER is still run as a cross-check against that label. Cases where they
  disagree are flagged - these are your likely sarcasm/nuance cases, same
  idea as the "rating vs VADER mismatch" logic from the original plan.
- Note: since there's no film identity in this dataset, there's no
  `film_slug` column anymore. The Step 6b film-level dashboard won't apply
  to this dataset version - only single-review analysis will work.
 
Run:
    python preprocess.py
"""
 
import re
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
 
# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
 
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # needed for newer nltk versions
 
INPUT_CSV = "../data/IMDB Dataset.csv"
OUTPUT_REVIEWS_CSV = "../data/reviews_labeled.csv"
OUTPUT_SENTENCES_CSV = "../data/sentences_labeled.csv"
 
analyzer = SentimentIntensityAnalyzer()
 
 
# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
 
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<br\s*/?>", " ", text)              # this dataset has literal <br /> tags
    text = re.sub(r"http\S+|www\.\S+", "", text)          # remove URLs
    text = re.sub(r"\s+", " ", text)                        # collapse whitespace
    text = text.strip()
    return text
 
 
# ---------------------------------------------------------------------------
# Sentiment cross-check
# ---------------------------------------------------------------------------
 
def vader_to_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"  # VADER can output neutral even though ground truth is binary
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} raw reviews")
    print("Columns found:", list(df.columns))
 
    df = df.rename(columns={"review": "review_text", "sentiment": "label_sentiment"})
    df["review_text"] = df["review_text"].apply(clean_text)
    df = df[df["review_text"].str.len() > 10].reset_index(drop=True)
    print(f"{len(df)} reviews remain after removing empty/very short entries")
 
    df["review_id"] = df.index  # stable id used to link sentences back later
 
    df["vader_label"] = df["review_text"].apply(vader_to_sentiment)
 
    # Ground truth is the dataset's own label - VADER is just a cross-check
    df["sentiment"] = df["label_sentiment"]
    df["mismatch"] = df["sentiment"] != df["vader_label"]
 
    mismatch_pct = df["mismatch"].mean() * 100
    print(f"Label/VADER mismatch rate: {mismatch_pct:.1f}% "
          f"(these are your likely sarcasm/nuance cases)")
 
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())
 
    df.to_csv(OUTPUT_REVIEWS_CSV, index=False)
    print(f"\nSaved review-level labels to {OUTPUT_REVIEWS_CSV}")
 
    # -----------------------------------------------------------------
    # Sentence-level split (for aspect extraction later)
    # -----------------------------------------------------------------
    sentence_rows = []
    for _, row in df.iterrows():
        sentences = nltk.sent_tokenize(row["review_text"])
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 4:  # skip junk fragments like "!" or ".."
                continue
            sentence_rows.append({
                "review_id": row["review_id"],
                "sentence": sent,
                "review_sentiment": row["sentiment"],  # useful as a weak prior
            })
 
    sentences_df = pd.DataFrame(sentence_rows)
    sentences_df.to_csv(OUTPUT_SENTENCES_CSV, index=False)
    print(f"Saved {len(sentences_df)} sentences to {OUTPUT_SENTENCES_CSV}")
 
 
if __name__ == "__main__":
    main()