"""
Letterboxd Review Scraper
--------------------------
Scrapes public review pages for a list of films and saves results to CSV.
 
Usage:
    python letterboxd_scraper.py
 
Notes:
- Letterboxd has no official API. This scrapes public HTML pages.
- Site structure can change over time — if fields come back empty,
  inspect the page HTML (right-click > Inspect on a review) and update
  the CSS selectors in `parse_review_block`.
- Be a good citizen: keep delays between requests, don't parallelize
  aggressively, and don't scrape more than you need for the project.
"""
 
import time
import random
import csv
import re
from datetime import datetime
 
import requests
from bs4 import BeautifulSoup
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
# Add / remove film slugs here. Slug = the part of the URL after /film/
# e.g. https://letterboxd.com/film/oppenheimer/ -> "oppenheimer"
FILM_SLUGS = [
    "oppenheimer",
    "barbie",
    "dune-part-two",
    "poor-things",
    "the-substance",
    "anatomy-of-a-fall",
    "past-lives",
    "the-holdovers",
    "killers-of-the-flower-moon",
    "spider-man-across-the-spider-verse",
]
 
MAX_PAGES_PER_FILM = 40   # ~12 reviews per page -> up to ~480 reviews/film
OUTPUT_CSV = "letterboxd_reviews.csv"
REQUEST_DELAY_RANGE = (1.0, 2.5)  # seconds, randomized delay between requests
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
 
BASE_URL = "https://letterboxd.com/film/{slug}/reviews/by/activity/page/{page}/"
 
 
# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
 
def star_string_to_rating(class_list):
    """
    Letterboxd encodes star ratings as a class like 'rated-8' on a <span>,
    where the number is half-stars (1-10 -> 0.5 to 5.0 stars).
    """
    for cls in class_list:
        match = re.match(r"rated-(\d+)", cls)
        if match:
            return int(match.group(1)) / 2.0
    return None
 
 
def parse_review_block(block, film_slug):
    """Extract fields from a single review <li> block."""
    review = {
        "film_slug": film_slug,
        "rating": None,
        "review_text": None,
        "date": None,
        "likes": None,
    }
 
    # --- Rating ---
    rating_span = block.select_one("span.rating")
    if rating_span:
        review["rating"] = star_string_to_rating(rating_span.get("class", []))
 
    # --- Review text ---
    body = block.select_one("div.body-text")
    if body:
        # Remove "This review may contain spoilers" type notices if present
        for tag in body.select("p.contains-spoilers-warning"):
            tag.decompose()
        paragraphs = [p.get_text(strip=True) for p in body.find_all("p")]
        review["review_text"] = " ".join(paragraphs).strip()
 
    # --- Date ---
    date_tag = block.select_one("span._nobr") or block.select_one("time")
    if date_tag:
        review["date"] = date_tag.get_text(strip=True)
 
    # --- Likes (optional, may not always be present in the HTML) ---
    likes_tag = block.select_one("p.like-link-target, span.like-link-target")
    if likes_tag:
        likes_text = likes_tag.get_text(strip=True)
        digits = re.sub(r"[^\d]", "", likes_text)
        review["likes"] = int(digits) if digits else 0
 
    return review
 
 
def fetch_page(slug, page_num):
    url = BASE_URL.format(slug=slug, page=page_num)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return None
    return resp.text
 
 
def scrape_film(slug, max_pages=MAX_PAGES_PER_FILM):
    """Scrape all review pages for a single film slug."""
    all_reviews = []
 
    for page_num in range(1, max_pages + 1):
        html = fetch_page(slug, page_num)
        if not html:
            print(f"  [{slug}] page {page_num}: request failed, stopping.")
            break
 
        soup = BeautifulSoup(html, "lxml")
        review_blocks = soup.select("li.film-detail")
 
        if not review_blocks:
            print(f"  [{slug}] page {page_num}: no more reviews, stopping.")
            break
 
        for block in review_blocks:
            review = parse_review_block(block, slug)
            if review["review_text"]:  # skip empty/ratings-only entries
                all_reviews.append(review)
 
        print(f"  [{slug}] page {page_num}: {len(review_blocks)} reviews found "
              f"(running total: {len(all_reviews)})")
 
        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
 
    return all_reviews
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    all_data = []
 
    for slug in FILM_SLUGS:
        print(f"Scraping film: {slug}")
        film_reviews = scrape_film(slug)
        all_data.extend(film_reviews)
        print(f"-> {len(film_reviews)} reviews collected for {slug}\n")
 
    if not all_data:
        print("No reviews collected. Check selectors / connectivity.")
        return
 
    fieldnames = ["film_slug", "rating", "review_text", "date", "likes"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
 
    print(f"Done. Saved {len(all_data)} total reviews to {OUTPUT_CSV}")
 
 
if __name__ == "__main__":
    main()