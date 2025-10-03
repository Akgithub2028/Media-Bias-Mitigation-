#!/usr/bin/env python3
# news-only_scraper.py
"""
Scrapes news articles matching event keywords from configured news sites.
Tweet scraping and snscrape dependency have been removed.
"""

# ==========================
# 1. Install dependencies
# ==========================
# !pip install requests beautifulsoup4 tqdm

# ==========================
# 2. (Optional) Mount Drive
# ==========================
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = "/content/drive/MyDrive/ideobias_data"
DATA_DIR = "/content/ideobias_data"   # local if no Drive
import os
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================
# 3. Imports
# ==========================
import os, re, json, time, requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from tqdm import tqdm

USER_AGENT = "Mozilla/5.0 (compatible; RawDataBot/1.0; +https://example.org/bot)"
HEADERS = {"User-Agent": USER_AGENT}

OUTDIR_NEWS = os.path.join(DATA_DIR, "raw_news")
os.makedirs(OUTDIR_NEWS, exist_ok=True)

# ==========================
# 4. Expanded Event Keywords
# ==========================
EVENT_KEYWORDS = {
    "demonetization": [
        "demonetisation","demonetization","note ban","currency ban","500 note","1000 note",
        "cash crunch","atm queue","withdrawal limit","currency exchange","digital payment",
        "upi","cashless economy","black money","currency switch","fake currency"
    ],
    "gst": [
        "gst","goods and services tax","goods service tax","indirect tax","gst rollout",
        "gst implementation","gst rates","gst council","gst slabs","gabbar singh tax",
        "gst compliance","gst returns","input tax credit"
    ],
    "farmers_protest": [
        "farmers protest","farmers' march","farmers agitation","farmers rally","farmer strike",
        "kisan andolan","farm bills protest","agrarian crisis","farm loan waiver",
        "minimum support price","msps","crop prices","agrarian distress"
    ],
    "migrant_crisis": [
        "migrant crisis","migrant workers","migrant labour","migrant labor","migrant exodus",
        "reverse migration","migrant walk","migrant lockdown","migrant deaths","migrant trains",
        "shramik trains","migrant relief","migrant shelters"
    ],
    "ladakh_protest": [
        "ladakh protest","ladakh standoff","ladakh border clash","galwan valley","ladakh tensions",
        "ladakh soldiers","china ladakh","ladakh disengagement","ladakh escalation",
        "ladakh skirmish","ladakh conflict"
    ]
}

# ==========================
# 5. News Sites Config
# ==========================
NEWS_SITES = {
    "timesofindia": {
        "label": "Times of India",
        "search_url": "https://timesofindia.indiatimes.com/topic/{q}",
        "host": "timesofindia.indiatimes.com"
    },
    "thehindu": {
        "label": "The Hindu",
        "search_url": "https://www.thehindu.com/search/?q={q}",
        "host": "www.thehindu.com"
    },
    "deccanherald": {
        "label": "Deccan Herald",
        "search_url": "https://www.deccanherald.com/search?q={q}",
        "host": "www.deccanherald.com"
    },
    "indianexpress": {
        "label": "Indian Express",
        "search_url": "https://indianexpress.com/?s={q}",
        "host": "indianexpress.com"
    },
    "hindustantimes": {
        "label": "Hindustan Times",
        "search_url": "https://www.hindustantimes.com/search?q={q}",
        "host": "www.hindustantimes.com"
    }
}

# ==========================
# 6. Helpers
# ==========================
def safe_get(url, retries=3, timeout=15):
    for _ in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.text
        except Exception:
            time.sleep(1)
    return None

def extract_article(html, url):
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.find("title"):
        title = soup.find("title").get_text().strip()
    elif soup.find("meta", {"property":"og:title"}):
        title = soup.find("meta", {"property":"og:title"}).get("content","").strip()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body = "\n".join([p for p in paragraphs if len(p) > 50])
    return {"url": url, "title": title, "text": body}

def collect_articles(site_key, keyword, max_links=30):
    site = NEWS_SITES[site_key]
    q = quote_plus(keyword)
    url = site["search_url"].format(q=q)
    html = safe_get(url)
    if not html: return []
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if site["host"] in href:
            if href.startswith("//"): href = "https:" + href
            if href.startswith("/"): href = "https://" + site["host"] + href
            links.add(href.split("?")[0])
    results = []
    for link in tqdm(list(links)[:max_links], desc=f"{site_key}-{keyword}"):
        h = safe_get(link)
        if not h: continue
        art = extract_article(h, link)
        if art["text"]:
            results.append(art)
    return results

def save_jsonl(path, items):
    with open(path, "w", encoding="utf8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

# ==========================
# 7. Main Runner (news only)
# ==========================
for event, keywords in EVENT_KEYWORDS.items():
    all_articles = []
    for kw in keywords:
        for site in NEWS_SITES.keys():
            try:
                arts = collect_articles(site, kw)
                for a in arts:
                    a["event"] = event
                    a["keyword"] = kw
                all_articles.extend(arts)
            except Exception as e:
                print(f"Error on {site}-{kw}: {e}")
    save_jsonl(os.path.join(OUTDIR_NEWS, f"{event}_articles.jsonl"), all_articles)

print("✅ News scraping complete. Data saved in:", OUTDIR_NEWS)
