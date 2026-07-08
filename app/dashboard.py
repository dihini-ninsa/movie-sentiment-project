"""
Step 6b: Streamlit Dashboard
---------------------------------
Interactive frontend for the project. Two modes:
    1. Single review analysis - paste a review, see aspect/sentiment breakdown
    2. Film-level dashboard - pick a film, see aggregated aspect sentiment
 
Run:
    streamlit run dashboard.py
 
Note: this calls the FastAPI backend, so make sure that's running first:
    uvicorn api:app --reload
"""
 
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
 
API_URL = "http://127.0.0.1:8000"
 
st.set_page_config(page_title="Movie Review Aspect Sentiment", layout="wide")
st.title("🎬 Movie Review Aspect-Based Sentiment Analysis")
 
mode = st.sidebar.radio("Choose a mode:", ["Single Review Analysis", "Film Dashboard"])
 
# ---------------------------------------------------------------------------
# Mode 1: Single review analysis
# ---------------------------------------------------------------------------
if mode == "Single Review Analysis":
    st.subheader("Analyze a single review")
 
    review_text = st.text_area(
        "Paste a movie review below:",
        height=150,
        placeholder="e.g. The acting was phenomenal but the pacing dragged in the second half...",
    )
 
    if st.button("Analyze"):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/analyze",
                    json={"review_text": review_text},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
 
                st.markdown(f"**Overall sentiment:** `{data['overall_sentiment'].upper()}`")
                st.markdown("---")
                st.markdown("**Sentence-by-sentence breakdown:**")
 
                breakdown_df = pd.DataFrame(data["breakdown"])
                st.dataframe(breakdown_df, use_container_width=True)
 
                color_map = {"positive": "green", "negative": "red", "neutral": "gray"}
                for row in data["breakdown"]:
                    color = color_map.get(row["sentiment"], "black")
                    st.markdown(
                        f"- **[{row['aspect']}]** "
                        f":{color}[{row['sentiment'].upper()}] — {row['sentence']}"
                    )
 
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the API. Make sure it's running: "
                    "`uvicorn api:app --reload`"
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
 
# ---------------------------------------------------------------------------
# Mode 2: Film-level dashboard
# ---------------------------------------------------------------------------
else:
    st.subheader("Film-level aspect sentiment summary")
 
    film_slug = st.text_input(
        "Enter a film slug (e.g. 'oppenheimer', matching your scraped data):"
    )
 
    if st.button("Load Film Summary"):
        if not film_slug.strip():
            st.warning("Please enter a film slug.")
        else:
            try:
                response = requests.get(f"{API_URL}/film/{film_slug}", timeout=10)
                response.raise_for_status()
                data = response.json()
 
                counts = data["aspect_sentiment_counts"]
                df = pd.DataFrame(counts).T.fillna(0)
 
                st.markdown(f"### Aspect breakdown for `{film_slug}`")
                st.dataframe(df, use_container_width=True)
 
                fig, ax = plt.subplots(figsize=(8, 5))
                df.plot(kind="bar", stacked=True, ax=ax)
                ax.set_ylabel("Number of sentences")
                ax.set_xlabel("Aspect")
                ax.set_title(f"Aspect Sentiment Breakdown: {film_slug}")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)
 
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the API. Make sure it's running: "
                    "`uvicorn api:app --reload`"
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")