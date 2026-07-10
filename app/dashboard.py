"""
Step 6b: Streamlit Dashboard
---------------------------------
Interactive frontend for the project. Two modes:
    1. Single Review Analysis - paste a review, see:
        - the review text with each sentence highlighted green/red by
          detected sentiment, tagged with its aspect
        - a per-aspect sentiment bar (green % vs red %) with thumbs
        - an overall sentiment gauge bar summarizing the whole review
    2. Overall Aspect Summary - aggregated aspect sentiment across the whole
       dataset (this dataset has no per-film identity, so this is dataset-wide
       rather than per-film)
 
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
st.markdown("## 🎬 Movie Review Aspect-Based Sentiment Analysis")
 
mode = st.sidebar.radio("Choose a mode:", ["Single Review Analysis", "Overall Aspect Summary"])
 
# Colors used throughout - kept consistent across the highlighted text,
# the per-aspect bars, and the overall gauge.
POSITIVE_COLOR = "#1f9d55"   # green
NEGATIVE_COLOR = "#d93f3f"   # red
POSITIVE_BG = "rgba(31, 157, 85, 0.28)"
NEGATIVE_BG = "rgba(217, 63, 63, 0.28)"
 
 
def render_highlighted_review(breakdown):
    """
    Renders the review text with each sentence highlighted by its detected
    sentiment (green=positive, red=negative), with a thumb icon and the
    aspect label shown as a small badge right after the sentence.
    """
    parts = []
    for row in breakdown:
        sentiment = row["sentiment"]
        bg = POSITIVE_BG if sentiment == "positive" else NEGATIVE_BG
        thumb = "👍" if sentiment == "positive" else "👎"
        aspect_badge = (
            f'<span style="font-size:0.7rem; opacity:0.75; '
            f'margin-left:0.3rem;">[{row["aspect"]}]</span>'
        )
        parts.append(
            f'<span style="background:{bg}; padding:0.15rem 0.35rem; '
            f'border-radius:5px; line-height:2.1;">{row["sentence"]}</span> '
            f'{thumb}{aspect_badge} '
        )
    html = "<div style='font-size:1.05rem; line-height:2.1;'>" + "".join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)
 
 
def render_aspect_bar(aspect, positive, negative):
    """Renders a single aspect's positive/negative split as an inline HTML bar."""
    total = positive + negative
    if total == 0:
        st.markdown(f"**{aspect}**: N/A")
        return
 
    pos_pct = round(100 * positive / total)
    neg_pct = 100 - pos_pct
 
    bar_html = f"""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.55rem;">
        <div style="width:150px; font-size:0.88rem;">{aspect}</div>
        <div style="flex:1; display:flex; height:22px; border-radius:5px; overflow:hidden;
                    background:#2a2a2a;">
            <div style="width:{pos_pct}%; background:{POSITIVE_COLOR};
                        display:flex; align-items:center; justify-content:flex-start;
                        padding-left:6px; color:white; font-size:0.72rem;">
                {pos_pct}% 
            </div>
            <div style="width:{neg_pct}%; background:{NEGATIVE_COLOR};"></div>
        </div>
        <div style="width:45px; text-align:right;">👍{"👎" if neg_pct > 0 else ""}</div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)
 
 
def render_overall_gauge(overall_sentiment, breakdown):
    """
    Renders a single gradient-style bar summarizing the whole review's
    sentiment, plus a Positive / Mixed-Neutral / Negative label based on
    the ratio of positive to negative sentences.
    """
    pos_count = sum(1 for r in breakdown if r["sentiment"] == "positive")
    neg_count = sum(1 for r in breakdown if r["sentiment"] == "negative")
    total = pos_count + neg_count
 
    if total == 0:
        label = "N/A"
        pos_pct = 50
    else:
        pos_pct = round(100 * pos_count / total)
        if pos_pct >= 65:
            label = "Positive"
        elif pos_pct <= 35:
            label = "Negative"
        else:
            label = "Mixed / Neutral"
 
    neg_pct = 100 - pos_pct
 
    st.markdown(f"### {label}")
    gauge_html = f"""
    <div style="display:flex; align-items:center; gap:0.8rem;">
        <div style="flex:1; display:flex; height:26px; border-radius:6px; overflow:hidden;
                    background:#2a2a2a;">
            <div style="width:{pos_pct}%; background:linear-gradient(90deg, {POSITIVE_COLOR}, #6fbf73);"></div>
            <div style="width:{neg_pct}%; background:linear-gradient(90deg, #e57373, {NEGATIVE_COLOR});"></div>
        </div>
        <div style="font-size:1.2rem;">👍/👎</div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)
 
 
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
                breakdown = data["breakdown"]
 
                st.markdown("#### Analysis Results")
 
                st.markdown("**Analyzed Review (with aspects highlighted):**")
                render_highlighted_review(breakdown)
 
                st.markdown("---")
                st.markdown("**Aspect-based Sentiment Breakdown:**")
 
                # Aggregate sentence-level results per aspect for this one review
                per_aspect = {}
                for row in breakdown:
                    a = row["aspect"]
                    per_aspect.setdefault(a, {"positive": 0, "negative": 0})
                    per_aspect[a][row["sentiment"]] += 1
 
                for aspect, counts in per_aspect.items():
                    render_aspect_bar(aspect, counts["positive"], counts["negative"])
 
                st.markdown("---")
                st.markdown("**Overall Review Sentiment:**")
                render_overall_gauge(data["overall_sentiment"], breakdown)
 
                with st.expander("See raw sentence-by-sentence table"):
                    st.dataframe(pd.DataFrame(breakdown), use_container_width=True)
 
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the API. Make sure it's running: "
                    "`uvicorn api:app --reload`"
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
 
# ---------------------------------------------------------------------------
# Mode 2: Overall aspect summary (dataset-wide - no per-film breakdown,
# since this dataset has no film identity column)
# ---------------------------------------------------------------------------
else:
    st.subheader("Overall aspect sentiment summary")
    st.caption(
        "This dataset (Kaggle IMDB 50K) has no per-film identity, so this shows "
        "aggregated aspect sentiment across all reviews rather than per-film."
    )
 
    if st.button("Load Summary"):
        try:
            response = requests.get(f"{API_URL}/aspects/summary", timeout=10)
            response.raise_for_status()
            data = response.json()
 
            counts = data["aspect_sentiment_counts"]
            df = pd.DataFrame(counts).T.fillna(0)
 
            st.markdown(f"### Aspect breakdown across {data['total_sentences']:,} sentences")
 
            for aspect, row in df.iterrows():
                render_aspect_bar(
                    aspect,
                    int(row.get("positive", 0)),
                    int(row.get("negative", 0)),
                )
 
            with st.expander("See raw counts table"):
                st.dataframe(df, use_container_width=True)
 
            fig, ax = plt.subplots(figsize=(8, 5))
            df.plot(kind="bar", stacked=True, ax=ax, color=[NEGATIVE_COLOR, POSITIVE_COLOR])
            ax.set_ylabel("Number of sentences")
            ax.set_xlabel("Aspect")
            ax.set_title("Aspect Sentiment Breakdown (Overall)")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
 
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the API. Make sure it's running: "
                "`uvicorn api:app --reload`"
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")