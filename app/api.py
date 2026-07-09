"""
Step 6a: FastAPI Backend
----------------------------
Serves the sentiment + aspect analysis as a REST API.
 
Endpoints:
    POST /analyze         -> analyze a single review, return aspect/sentiment breakdown
    GET  /aspects/summary  -> return aggregated aspect sentiment stats across
                              the whole dataset (no per-film breakdown - this
                              dataset has no film identity, see preprocess.py notes)
 
Run:
    uvicorn api:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""
 
import nltk
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
 
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
 
app = FastAPI(title="Movie Review Aspect-Sentiment API")
 
MODEL_PATH = "../data/baseline_model.joblib"
VECTORIZER_PATH = "../data/tfidf_vectorizer.joblib"
FINAL_DATA_PATH = "../data/aspect_sentiment_final.csv"
 
# Load models once at startup rather than per-request
clf = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
 
# This lets sentence-level results get grouped into approximate aspects
# based on simple keyword matching, as a lightweight companion to the
# full BERTopic pipeline for live single-sentence requests.
ASPECT_KEYWORDS = {
    "Acting": ["acting", "performance", "actor", "actress", "cast"],
    "Plot/Story": ["plot", "story", "storyline", "narrative"],
    "Cinematography": ["cinematography", "visuals", "shot", "camera", "framing"],
    "Pacing": ["pacing", "pace", "slow", "dragged", "rushed"],
    "Music/Score": ["music", "score", "soundtrack"],
    "Ending": ["ending", "finale", "conclusion"],
    "Direction": ["direction", "directing", "director"],
    "Dialogue/Writing": ["dialogue", "writing", "script", "lines"],
}
 
 
def guess_aspect(sentence):
    sentence_lower = sentence.lower()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in sentence_lower for kw in keywords):
            return aspect
    return "General"
 
 
class ReviewRequest(BaseModel):
    review_text: str
 
 
class SentenceResult(BaseModel):
    sentence: str
    aspect: str
    sentiment: str
 
 
class ReviewResponse(BaseModel):
    overall_sentiment: str
    breakdown: list[SentenceResult]
 
 
@app.post("/analyze", response_model=ReviewResponse)
def analyze_review(payload: ReviewRequest):
    text = payload.review_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="review_text cannot be empty")
 
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        raise HTTPException(status_code=400, detail="Could not extract sentences from input")
 
    X = vectorizer.transform(sentences)
    preds = clf.predict(X)
 
    breakdown = [
        SentenceResult(sentence=sent, aspect=guess_aspect(sent), sentiment=pred)
        for sent, pred in zip(sentences, preds)
    ]
 
    overall_X = vectorizer.transform([text])
    overall_sentiment = clf.predict(overall_X)[0]
 
    return ReviewResponse(overall_sentiment=overall_sentiment, breakdown=breakdown)
 
 
@app.get("/aspects/summary")
def aspects_summary():
    try:
        df = pd.read_csv(FINAL_DATA_PATH)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Aspect data not found - run aspect_bertopic.py first",
        )
 
    summary = (
        df.groupby(["aspect", "sentence_sentiment"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    return {"total_sentences": len(df), "aspect_sentiment_counts": summary}
 
 
@app.get("/")
def root():
    return {"status": "ok", "message": "Movie Review Aspect-Sentiment API is running"}