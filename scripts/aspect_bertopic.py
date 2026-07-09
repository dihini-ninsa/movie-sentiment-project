"""
Step 5: BERTopic Aspect Discovery + Sentence-Level Sentiment
----------------------------------------------------------------
This is the core differentiator of the project: instead of hardcoding
aspects ("acting", "plot", "pacing"...), we let BERTopic discover them
directly from the sentence data.
 
Pipeline:
1. Embed every sentence with a sentence-transformer model
2. Cluster embeddings with BERTopic (UMAP + HDBSCAN under the hood)
3. Auto-map each discovered topic to a clean aspect name by matching its
   top keywords against ASPECT_KEYWORDS below. This is deliberately NOT
   based on topic ID numbers - BERTopic's topic numbering can shift
   slightly between runs (UMAP/HDBSCAN have residual non-determinism even
   with a fixed random_state under multi-threaded execution), so matching
   on keyword content instead of topic ID makes the mapping stable across
   runs, sample sizes, and even after upgrading the umap/bertopic packages.
4. Run sentiment on each individual sentence (reusing the LoRA model or the
   baseline model) to get per-sentence sentiment
5. Save a final (aspect, sentiment) table for the dashboard
 
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
 
# ---------------------------------------------------------------------------
# Aspect keyword sets, built from what showed up in real discovered topics
# across multiple runs. A topic gets matched to whichever aspect its top
# keywords overlap with most. Topics with no meaningful overlap (genre
# clusters like "horror"/"bollywood", filler like "oh yeah", DVD/rating
# mentions, etc.) are correctly left unmapped and dropped.
# ---------------------------------------------------------------------------
ASPECT_KEYWORDS = {
    "Acting": {"acting", "actor", "actress", "actors", "cast", "performance",
               "performances", "role", "supporting", "hes", "shes"},
    "Comedy/Humor": {"funny", "comedy", "laugh", "laughed", "laughing",
                      "jokes", "joke", "humor", "humour", "hilarious"},
    "Music/Score": {"music", "soundtrack", "song", "songs", "score",
                     "musical", "dance", "dancing", "sound"},
    "Plot/Story": {"plot", "story", "stories", "storyline", "plots",
                    "subplots", "narrative", "holes"},
    "Ending": {"ending", "end", "ends", "twist", "climax", "finale"},
    "Characters": {"characters", "character", "believable"},
    "Visual Effects/Cinematography": {"effects", "animation", "camera",
                                        "special", "cgi", "cinematography",
                                        "shots", "shot", "photography", "visual"},
    "Dialogue/Writing": {"script", "scripts", "dialogue", "written", "writing",
                          "direction"},
    "Pacing": {"minutes", "slow", "hours", "hour", "short", "long", "fast",
               "pace", "pacing", "length"},
}
 
 
def discover_topics(sentences):
    print("Loading sentence embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
 
    print(f"Embedding {len(sentences)} sentences...")
    embeddings = embedder.encode(sentences, show_progress_bar=True)
 
    print("Running BERTopic clustering...")
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
    info = topic_model.get_topic_info()
    print("\nDiscovered topics (excluding -1 = outliers/noise):")
    print(info[info["Topic"] != -1].head(top_n).to_string(index=False))
    return info
 
 
def auto_map_topics(topic_model, min_overlap=2):
    """
    Matches each discovered topic to an aspect by keyword overlap, instead
    of relying on topic ID numbers (which can shift between runs).
    Returns a dict {topic_id: aspect_name}, and prints the mapping so you
    can eyeball it before it's applied.
    """
    mapping = {}
    topic_info = topic_model.get_topic_info()
 
    print("\nAuto-mapped topics -> aspects (keyword-based, not ID-based):")
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            continue
 
        topic_words = {word for word, _ in topic_model.get_topic(topic_id)}
 
        best_aspect, best_overlap = None, 0
        for aspect, aspect_keywords in ASPECT_KEYWORDS.items():
            overlap = len(topic_words & aspect_keywords)
            if overlap > best_overlap:
                best_aspect, best_overlap = aspect, overlap
 
        if best_overlap >= min_overlap:
            mapping[topic_id] = best_aspect
            print(f"  Topic {topic_id} -> {best_aspect}  "
                  f"(matched on: {sorted(topic_words & ASPECT_KEYWORDS[best_aspect])})")
        else:
            print(f"  Topic {topic_id} -> [unmapped/dropped]  "
                  f"(top words: {sorted(topic_words)[:5]})")
 
    return mapping
 
 
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
 
    topic_model.save(BERTOPIC_MODEL_DIR, serialization="safetensors")
    print(f"Saved BERTopic model to {BERTOPIC_MODEL_DIR}")
 
    topic_to_aspect = auto_map_topics(topic_model)
 
    df["aspect"] = df["topic_id"].map(topic_to_aspect)
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