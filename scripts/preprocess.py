"""
Step 2: Preprocessing & Sentiment Labeling
--------------------------------------------
Takes the raw scraped CSV and produces two outputs:
 
1. reviews_labeled.csv      -> one row per review, with a sentiment label
2. sentences_labeled.csv    -> one row per sentence (needed later for BERTopic
                                aspect extraction, since aspects live at
                                sentence level, not whole-review level)
 
Sentiment labeling logic:
- Convert star rating (0.5 - 5.0) into a "rating-based" sentiment
- Run VADER on the review text for a "text-based" sentiment
- Combine both: if they agree, use that label. If they disagree, trust the
  rating (since it's ground truth from the user) but flag it in a column
  so you can inspect mismatches later (useful for your project write-up,
  e.g. "X% of reviews had sarcasm/mismatch between rating and text sentiment")
 
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
 
INPUT_CSV = "../data/letterboxd_reviews.csv"
OUTPUT_REVIEWS_CSV = "../data/reviews_labeled.csv"
OUTPUT_SENTENCES_CSV = "../data/sentences_labeled.csv"
 
analyzer = SentimentIntensityAnalyzer()
 
 
# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
 
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)      # remove URLs
    text = re.sub(r"\s+", " ", text)                    # collapse whitespace
    text = text.strip()
    return text
 
 
# ---------------------------------------------------------------------------
# Sentiment labeling
# ---------------------------------------------------------------------------
 
def rating_to_sentiment(rating):
    """Letterboxd ratings run 0.5 - 5.0 in half-star steps."""
    if rating is None or pd.isna(rating):
        return None
    if rating >= 3.5:
        return "positive"
    elif rating <= 2.5:
        return "negative"
    else:
        return "neutral"
 
 
def vader_to_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"
 
 
def combine_labels(rating_label, vader_label):
    """
    Trust the star rating as ground truth (it's the user's explicit signal).
    VADER is used as a cross-check -> mismatches are flagged, not overridden.
    If rating is missing, fall back to VADER.
    """
    if rating_label is None:
        return vader_label, False
    mismatch = rating_label != vader_label
    return rating_label, mismatch
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} raw reviews")
 
    df["review_text"] = df["review_text"].apply(clean_text)
    df = df[df["review_text"].str.len() > 10].reset_index(drop=True)
    print(f"{len(df)} reviews remain after removing empty/very short entries")
 
    df["review_id"] = df.index  # stable id used to link sentences back later
 
    rating_labels = df["rating"].apply(rating_to_sentiment)
    vader_labels = df["review_text"].apply(vader_to_sentiment)
 
    final_labels = []
    mismatches = []
    for r_label, v_label in zip(rating_labels, vader_labels):
        label, mismatch = combine_labels(r_label, v_label)
        final_labels.append(label)
        mismatches.append(mismatch)
 
    df["vader_label"] = vader_labels
    df["sentiment"] = final_labels
    df["rating_vader_mismatch"] = mismatches
 
    mismatch_pct = df["rating_vader_mismatch"].mean() * 100
    print(f"Rating/VADER mismatch rate: {mismatch_pct:.1f}% "
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
                "film_slug": row["film_slug"],
                "sentence": sent,
                "review_sentiment": row["sentiment"],  # useful as a weak prior
            })
 
    sentences_df = pd.DataFrame(sentence_rows)
    sentences_df.to_csv(OUTPUT_SENTENCES_CSV, index=False)
    print(f"Saved {len(sentences_df)} sentences to {OUTPUT_SENTENCES_CSV}")
 
 
if __name__ == "__main__":
    main()