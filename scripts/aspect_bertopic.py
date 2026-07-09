"""
Step 5: BERTopic Aspect Discovery + Sentence-Level Sentiment
----------------------------------------------------------------
This is the core differentiator of the project: instead of hardcoding
aspects ("acting", "plot", "pacing"...), we let BERTopic discover them
directly from the sentence data.
 
Pipeline:
1. Embed every sentence with a sentence-transformer model
2. Cluster embeddings with BERTopic (UMAP + HDBSCAN under the hood)
3. Inspect discovered topics, manually map raw topic IDs -> clean aspect
   names (this step needs a human - BERTopic gives you clusters + keywords,
   you decide what to call them and which to merge)
4. Run sentiment on each individual sentence (reusing the LoRA model or the
   baseline model) to get per-sentence sentiment
5. Save a final (film, aspect, sentiment) table for the dashboard
 
Run:
    python aspect_bertopic.py
"""
 
import pandas as pd
import joblib
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
 
INPUT_SENTENCES_CSV = "../data/sentences_labeled.csv"
 
# Set to a number (e.g. 50000) to test on a subset first - much faster.
# Set to None to run on the full dataset (612k sentences - will take a while on CPU).
SAMPLE_SIZE = 50000
OUTPUT_TOPICS_CSV = "../data/discovered_topics.csv"
OUTPUT_FINAL_CSV = "../data/aspect_sentiment_final.csv"
BERTOPIC_MODEL_DIR = "../data/bertopic_model"
 
# Baseline model used here for simplicity/speed. Swap in your LoRA model
# (load with peft.PeftModel.from_pretrained) if you want the fancier version.
BASELINE_MODEL_PATH = "../data/baseline_model.joblib"
VECTORIZER_PATH = "../data/tfidf_vectorizer.joblib"
 
 
def discover_topics(sentences):
    print("Loading sentence embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
 
    print(f"Embedding {len(sentences)} sentences...")
    embeddings = embedder.encode(sentences, show_progress_bar=True)
 
    print("Running BERTopic clustering...")
    # random_state is set on UMAP so topic numbers stay stable across runs -
    # without this, re-running produces a DIFFERENT topic numbering each time,
    # which silently breaks any TOPIC_TO_ASPECT mapping written for a prior run.
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        min_topic_size=30,     # smaller = more granular topics, more noise
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(sentences, embeddings)
 
    return topic_model, topics
 
 
def print_topic_summary(topic_model, top_n=25):
    """
    Prints the discovered topics with their top keywords so you can manually
    decide which raw topic IDs correspond to which clean aspect names.
    """
    info = topic_model.get_topic_info()
    print("\nDiscovered topics (excluding -1 = outliers/noise):")
    print(info[info["Topic"] != -1].head(top_n).to_string(index=False))
    return info
 
 
# ---------------------------------------------------------------------------
# MANUAL STEP: after running once and inspecting printed topics, fill this
# mapping in based on what you see, then re-run with SKIP_DISCOVERY = True
# to skip re-clustering and just apply your mapping.
#
# Example (topic IDs will differ for your actual data):
# TOPIC_TO_ASPECT = {
#     0: "Acting",
#     1: "Plot/Story",
#     2: "Cinematography",
#     3: "Pacing",
#     4: "Music/Score",
#     5: "Ending",
#     6: "Direction",
#     7: "Dialogue/Writing",
# }
# ---------------------------------------------------------------------------
TOPIC_TO_ASPECT = {
    0: "Acting",              # her, she, shes, actress
    4: "Acting",              # acting, cast, actors, supporting
    1: "Comedy/Humor",        # funny, comedy, laugh, jokes
    2: "Music/Score",         # music, soundtrack, song, songs
    5: "Plot/Story",          # plot, story, stories, plots
    9: "Ending",              # ending, end, twist, climax
    11: "Characters",         # characters, character, are, were
    12: "Visual Effects/Cinematography",  # effects, animation, camera, special
    15: "Dialogue/Writing",   # script, acting, bad, scripts
    17: "Pacing",             # minutes, slow, hours, short
}
 
 
def apply_sentiment(sentences_df):
    """Applies the baseline model to label sentiment per sentence."""
    clf = joblib.load(BASELINE_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
 
    X = vectorizer.transform(sentences_df["sentence"])
    sentences_df["sentence_sentiment"] = clf.predict(X)
    return sentences_df
 
 
def main():
    df = pd.read_csv(INPUT_SENTENCES_CSV)
    df = df.dropna(subset=["sentence"])
 
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"Using a random sample of {SAMPLE_SIZE} sentences "
              f"(set SAMPLE_SIZE = None at the top to run on the full dataset)")
 
    sentences = df["sentence"].tolist()
 
    topic_model, topics = discover_topics(sentences)
    df["topic_id"] = topics
 
    topic_info = print_topic_summary(topic_model)
    topic_info.to_csv(OUTPUT_TOPICS_CSV, index=False)
    print(f"\nSaved full topic list to {OUTPUT_TOPICS_CSV}")
    print("\n--> Open that CSV, inspect the keywords per topic, then fill in "
          "TOPIC_TO_ASPECT at the top of this script with your own labels.")
 
    topic_model.save(BERTOPIC_MODEL_DIR, serialization="safetensors")
    print(f"Saved BERTopic model to {BERTOPIC_MODEL_DIR}")
 
    if not TOPIC_TO_ASPECT:
        print("\nTOPIC_TO_ASPECT is empty - stopping here so you can fill it "
              "in based on the printed topics. Re-run after editing.")
        return
 
    df["aspect"] = df["topic_id"].map(TOPIC_TO_ASPECT)
    df = df.dropna(subset=["aspect"])  # drops sentences in unmapped/noise topics
 
    print("\nApplying sentiment model to each sentence...")
    df = apply_sentiment(df)
 
    final_cols = ["review_id", "sentence", "aspect", "sentence_sentiment"]
    df[final_cols].to_csv(OUTPUT_FINAL_CSV, index=False)
    print(f"\nSaved final aspect+sentiment table to {OUTPUT_FINAL_CSV}")
 
    print("\nAspect x Sentiment breakdown:")
    print(pd.crosstab(df["aspect"], df["sentence_sentiment"]))
 
 
if __name__ == "__main__":
    main()