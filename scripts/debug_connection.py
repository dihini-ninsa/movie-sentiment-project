"""
Diagnostic script - run this to see exactly why requests are failing.
"""
 
import requests
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
 
url = "https://letterboxd.com/film/oppenheimer/reviews/by/activity/page/1/"
 
try:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    print("Status code:", resp.status_code)
    print("Response headers:", dict(resp.headers))
    print("\nFirst 500 chars of response body:")
    print(resp.text[:500])
except requests.exceptions.RequestException as e:
    print("Request raised an exception:")
    print(type(e).__name__, "-", str(e))
 