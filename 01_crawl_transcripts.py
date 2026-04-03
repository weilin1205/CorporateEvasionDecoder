"""
01_crawl_transcripts.py
Fetch earnings-call transcripts from SeekingAlpha via RapidAPI.

API:  seeking-alpha.p.rapidapi.com
      /transcripts/v2/list?id={ticker}&size=20&number=1
      /transcripts/v2/get-details?id={article_id}

Stores raw JSON (with extracted plain text) under data/raw_transcripts/<TICKER>/.

Usage:
    python 01_crawl_transcripts.py
"""
import http.client
import json
import os
import time
import logging

import pandas as pd
from bs4 import BeautifulSoup

from config import (
    RAPIDAPI_KEY, RAPIDAPI_HOST, MAX_API_CALLS,
    TRANSCRIPTS_PER_TICKER, RAW_TRANSCRIPT_DIR, TICKERS_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST,
    "Content-Type": "application/json",
}

api_call_count = 0


def api_get(path: str, retries: int = 3) -> dict:
    global api_call_count
    if api_call_count >= MAX_API_CALLS:
        raise RuntimeError(f"API call limit reached ({MAX_API_CALLS})")
    for attempt in range(retries):
        try:
            conn = http.client.HTTPSConnection(RAPIDAPI_HOST)
            conn.request("GET", path, headers=HEADERS)
            res = conn.getresponse()
            raw = res.read()
            conn.close()
            api_call_count += 1
            decoded = raw.decode("utf-8").strip()
            if not decoded:
                raise ValueError(f"Empty response (HTTP {res.status})")
            return json.loads(decoded)
        except (json.JSONDecodeError, ValueError) as exc:
            wait = 2 ** (attempt + 1)
            logger.warning("API error on attempt %d/%d: %s — waiting %ds",
                           attempt + 1, retries, exc, wait)
            time.sleep(wait)
    logger.error("All %d retries failed for %s", retries, path)
    return {}


def get_transcript_list(ticker: str, page: int = 1, size: int = 20) -> list:
    path = f"/transcripts/v2/list?id={ticker}&size={size}&number={page}"
    logger.info("List %s page=%d  [API #%d]", ticker, page, api_call_count + 1)
    result = api_get(path)
    if "data" not in result:
        logger.warning("No 'data' for %s: %s", ticker, list(result.keys()))
        return []
    return result["data"]


def get_transcript_details(article_id: str) -> dict:
    path = f"/transcripts/v2/get-details?id={article_id}"
    return api_get(path)


def extract_text_from_html(html: str) -> str:
    """Convert HTML transcript to plain text, preserving paragraph structure."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["span", "a", "strong", "br"]):
        tag.unwrap()
    paragraphs = soup.find_all("p")
    return "\n\n".join(p.get_text(strip=True) for p in paragraphs)


def crawl_all():
    global api_call_count
    df = pd.read_csv(TICKERS_PATH)
    total_saved = 0

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        sector = row.get("Sector", "")
        ticker_dir = os.path.join(RAW_TRANSCRIPT_DIR, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        existing_ids = {
            f.replace(".json", "") for f in os.listdir(ticker_dir) if f.endswith(".json")
        }

        try:
            items = get_transcript_list(ticker)
        except RuntimeError:
            logger.error("API limit hit — stopping.")
            break
        time.sleep(1.0)

        if not items:
            logger.warning("%s: 0 items returned", ticker)
            continue

        # Filter for earnings call transcripts
        earnings = [
            x for x in items
            if x.get("type") == "transcript"
            and any(kw in (x.get("attributes") or {}).get("title", "").lower()
                    for kw in ["earnings call", "earnings conference",
                               "results conference", "results -",
                               "quarterly results", "financial results"])
        ]
        if not earnings:
            # Fallback: take all transcripts
            earnings = [x for x in items if x.get("type") == "transcript"]

        to_fetch = [
            x for x in earnings[:TRANSCRIPTS_PER_TICKER]
            if str(x["id"]) not in existing_ids
        ]
        logger.info("%s [%s]: %d earnings, %d to fetch", ticker, sector, len(earnings), len(to_fetch))

        for item in to_fetch:
            tid = str(item["id"])
            try:
                details = get_transcript_details(tid)
                time.sleep(1.0)

                attrs = item.get("attributes") or {}
                det_data = details.get("data") or {}
                det_attrs = det_data.get("attributes") or {}

                html = det_attrs.get("content", "")
                content_text = extract_text_from_html(html) if html else ""

                record = {
                    "id": tid,
                    "ticker": ticker,
                    "sector": sector,
                    "title": attrs.get("title", ""),
                    "publishOn": attrs.get("publishOn", ""),
                    "content_text": content_text,
                    "content_html": html,
                    "source": "seeking-alpha-api",
                }

                out_path = os.path.join(ticker_dir, f"{tid}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                total_saved += 1
                logger.info("  Saved %s/%s  [API #%d]", ticker, tid, api_call_count)
            except RuntimeError:
                logger.error("API limit hit — stopping.")
                break
            except Exception as exc:
                logger.error("  Error %s/%s: %s", ticker, tid, exc)

        if api_call_count >= MAX_API_CALLS:
            break

    logger.info("Done — API calls: %d/%d, Saved: %d", api_call_count, MAX_API_CALLS, total_saved)


if __name__ == "__main__":
    crawl_all()
