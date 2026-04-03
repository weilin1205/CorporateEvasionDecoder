"""
02_extract_qa_pairs.py
Parse transcripts and extract structured Q&A pairs.

Supports TWO formats:
  1. Structured HTML (new API) — uses div classes like transcript-qa-section
  2. Plain text (old imported) — uses speaker name heuristics

Output: data/qa_pairs.json

Usage:
    python 02_extract_qa_pairs.py
"""
import json
import os
import re
import glob
import logging
from bs4 import BeautifulSoup
from config import RAW_TRANSCRIPT_DIR, QA_PAIRS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Method 1: Parse structured HTML (new API transcripts)
# ═══════════════════════════════════════════════════════════════════════════

def parse_html_participants(soup) -> tuple[set, set]:
    """Extract executive and analyst names from structured HTML."""
    executives: set[str] = set()
    analysts: set[str] = set()

    cp_div = soup.find("div", class_="transcript-company-participants")
    if cp_div:
        for span in cp_div.find_all("span"):
            text = span.get_text(strip=True)
            name = re.split(r"\s*[-–—]\s*", text)[0].strip()
            if name and len(name) > 2:
                executives.add(name.lower())
                parts = name.split()
                if len(parts) >= 2:
                    executives.add(parts[-1].lower())

    cc_div = soup.find("div", class_="transcript-other-participants")
    if cc_div:
        for span in cc_div.find_all("span"):
            text = span.get_text(strip=True)
            name = re.split(r"\s*[-–—]\s*", text)[0].strip()
            if name and len(name) > 2:
                analysts.add(name.lower())
                parts = name.split()
                if len(parts) >= 2:
                    analysts.add(parts[-1].lower())

    return executives, analysts


def classify_speaker(speaker: str, exec_names: set, analyst_names: set) -> str:
    sp = speaker.lower().strip()
    if sp == "operator":
        return "operator"
    for name in exec_names:
        if name in sp or sp in name:
            return "executive"
    for name in analyst_names:
        if name in sp or sp in name:
            return "analyst"
    return "unknown"


def extract_section_segments(soup, section_class: str) -> list[dict]:
    """Extract speaker segments from a structured div section (presentation or qa)."""
    section = soup.find("div", class_=section_class)
    if not section:
        return []

    segments = []
    subsections = section.find_all("div", class_=re.compile(r"section"), recursive=True)

    for sub in subsections:
        title_p = sub.find("p", class_=re.compile(r"title"))
        if not title_p:
            continue

        # Extract speaker name from the title (first <strong> or first line)
        strong = title_p.find("strong")
        speaker = strong.get_text(strip=True) if strong else title_p.get_text(strip=True).split("\n")[0].strip()

        # Extract speech from non-title <p> tags
        speech_parts = []
        for p in sub.find_all("p"):
            p_class = " ".join(p.get("class") or [])
            if "title" in p_class or "separator" in p_class:
                continue
            text = p.get_text(strip=True)
            if text:
                speech_parts.append(text)

        if speaker and speech_parts:
            segments.append({
                "speaker": speaker,
                "text": " ".join(speech_parts),
            })

    return segments


def extract_qa_from_html(record: dict) -> list[dict]:
    """Extract Q&A pairs from structured HTML transcript."""
    html = record.get("content_html", "")
    if not html or "structured-transcript" not in html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    exec_names, analyst_names = parse_html_participants(soup)

    if not exec_names and not analyst_names:
        return []

    # Try to find Q&A section div
    qa_segments = extract_section_segments(soup, "transcript-qa")

    # If no dedicated Q&A div, get ALL sections and find Q&A boundary
    if not qa_segments:
        all_segments = []
        for div in soup.find_all("div", class_=re.compile(r"transcript-(presentation|qa)")):
            segs = extract_section_segments_from_div(div, exec_names, analyst_names)
            all_segments.extend(segs)

        if not all_segments:
            # Fallback: parse all section divs
            all_segments = extract_all_section_divs(soup)

        # Find Q&A start
        qa_start = 0
        for i, seg in enumerate(all_segments):
            if "question" in seg.get("text", "").lower()[:200] and seg["speaker"].lower() == "operator":
                qa_start = i
                break
            if classify_speaker(seg["speaker"], exec_names, analyst_names) == "analyst":
                qa_start = i
                break

        qa_segments = all_segments[qa_start:]

    return pair_qa_segments(qa_segments, exec_names, analyst_names, record)


def extract_section_segments_from_div(div, exec_names, analyst_names) -> list[dict]:
    """Extract segments from a parent div."""
    segments = []
    for sub in div.find_all("div", recursive=False):
        cls = " ".join(sub.get("class") or [])
        if "section" not in cls:
            continue
        title_p = sub.find("p", class_=re.compile(r"title"))
        if not title_p:
            continue
        strong = title_p.find("strong")
        speaker = strong.get_text(strip=True) if strong else title_p.get_text(strip=True).split("\n")[0].strip()
        speech_parts = []
        for p in sub.find_all("p"):
            p_class = " ".join(p.get("class") or [])
            if "title" in p_class or "separator" in p_class:
                continue
            text = p.get_text(strip=True)
            if text:
                speech_parts.append(text)
        if speaker and speech_parts:
            segments.append({"speaker": speaker, "text": " ".join(speech_parts)})
    return segments


def extract_all_section_divs(soup) -> list[dict]:
    """Fallback: find all divs with 'section' in class that have a title."""
    segments = []
    for sub in soup.find_all("div", class_=re.compile(r"section")):
        if sub.find("div", class_=re.compile(r"section")):
            continue  # skip parent divs
        title_p = sub.find("p", class_=re.compile(r"title"))
        if not title_p:
            continue
        strong = title_p.find("strong")
        speaker = strong.get_text(strip=True) if strong else title_p.get_text(strip=True).split("\n")[0].strip()
        speech_parts = []
        for p in sub.find_all("p"):
            p_class = " ".join(p.get("class") or [])
            if "title" in p_class or "separator" in p_class:
                continue
            text = p.get_text(strip=True)
            if text:
                speech_parts.append(text)
        if speaker and speech_parts:
            segments.append({"speaker": speaker, "text": " ".join(speech_parts)})
    return segments


# ═══════════════════════════════════════════════════════════════════════════
# Method 2: Parse plain text (old imported transcripts)
# ═══════════════════════════════════════════════════════════════════════════

def parse_text_participants(text: str) -> tuple[set, set]:
    """Extract names from plain text using Name - Title pattern."""
    executives: set[str] = set()
    analysts: set[str] = set()

    # Find Company Participants section
    cp_match = re.search(
        r"Company Participants?\s*\n(.*?)(?:\n\s*\n|\nConference Call Participants?)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if cp_match:
        raw = cp_match.group(1).strip()
        # Handle concatenated names: split by Name - Title pattern
        names = re.findall(r"([A-Z][a-zA-Z'.]+(?:\s+[A-Z][a-zA-Z'.]+)*)\s*[-–—]\s*", raw)
        for name in names:
            name = name.strip()
            if name and len(name) > 2:
                executives.add(name.lower())
                parts = name.split()
                if len(parts) >= 2:
                    executives.add(parts[-1].lower())

    # Find Conference Call Participants section
    cc_match = re.search(
        r"Conference Call Participants?\s*\n(.*?)(?:\n\s*\n|\nPresentation|\nOperator)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if cc_match:
        raw = cc_match.group(1).strip()
        names = re.findall(r"([A-Z][a-zA-Z'.]+(?:\s+[A-Z][a-zA-Z'.]+)*)\s*[-–—]\s*", raw)
        for name in names:
            name = name.strip()
            if name and len(name) > 2:
                analysts.add(name.lower())
                parts = name.split()
                if len(parts) >= 2:
                    analysts.add(parts[-1].lower())

    return executives, analysts


def find_qa_start_text(text: str) -> int:
    markers = ["Question-and-Answer Session", "Question and Answer Session", "Q&A Session"]
    for m in markers:
        idx = text.find(m)
        if idx != -1:
            return idx + len(m)
    match = re.search(r"(?:our first question|open the call to questions)", text, re.IGNORECASE)
    if match:
        return text.rfind("\n\n", 0, match.start())
    return -1


def split_speaker_segments_text(text: str) -> list[dict]:
    blocks = re.split(r"\n\n+", text.strip())
    segments = []
    current_speaker = None
    current_parts = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        first_line = lines[0].strip()

        is_speaker = (
            len(first_line.split()) <= 6
            and not first_line.endswith((".", "?", "!", ",", ";", ":"))
            and not first_line.startswith(("--", "\"", "'", "(", "["))
            and len(first_line) < 80
            and not any(kw in first_line.lower() for kw in [
                "thank you", "thanks", "yes", "no ", "sure", "okay",
                "well ", "so ", "and ", "but ", "i ", "we ", "our ", "the ",
                "good ", "great ", "hi ", "hey ", "let me", "if you",
            ])
            and len(lines) == 1
        )

        if is_speaker:
            if current_speaker and current_parts:
                segments.append({"speaker": current_speaker, "text": " ".join(current_parts).strip()})
            current_speaker = first_line
            current_parts = []
        else:
            current_parts.append(block)

    if current_speaker and current_parts:
        segments.append({"speaker": current_speaker, "text": " ".join(current_parts).strip()})
    return segments


def extract_qa_from_text(record: dict) -> list[dict]:
    """Extract Q&A pairs from plain text transcript."""
    content = record.get("content_text", "") or record.get("content", "")
    if not content:
        return []

    exec_names, analyst_names = parse_text_participants(content)
    qa_start = find_qa_start_text(content)
    if qa_start == -1:
        return []

    qa_text = content[qa_start:]
    segments = split_speaker_segments_text(qa_text)
    if not segments:
        return []

    return pair_qa_segments(segments, exec_names, analyst_names, record)


# ═══════════════════════════════════════════════════════════════════════════
# Shared: pair analyst questions with executive answers
# ═══════════════════════════════════════════════════════════════════════════

def pair_qa_segments(segments, exec_names, analyst_names, record) -> list[dict]:
    """Match analyst questions with executive answers."""
    pairs = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        role = classify_speaker(seg["speaker"], exec_names, analyst_names)

        if role == "operator":
            i += 1
            continue

        if role == "analyst":
            question_speaker = seg["speaker"]
            question_parts = [seg["text"]]
            j = i + 1

            # Collect contiguous speech from same analyst
            while j < len(segments):
                r = classify_speaker(segments[j]["speaker"], exec_names, analyst_names)
                if r == "analyst" and segments[j]["speaker"] == question_speaker:
                    question_parts.append(segments[j]["text"])
                    j += 1
                else:
                    break

            # Collect executive answer
            answer_parts = []
            answer_speaker = ""
            while j < len(segments):
                r = classify_speaker(segments[j]["speaker"], exec_names, analyst_names)
                if r == "executive" or r == "unknown":
                    if not answer_speaker:
                        answer_speaker = segments[j]["speaker"]
                    answer_parts.append(segments[j]["text"])
                    j += 1
                elif r == "operator" or r == "analyst":
                    break
                else:
                    j += 1

            if question_parts and answer_parts:
                q = " ".join(question_parts)
                a = " ".join(answer_parts)
                if len(q.split()) >= 8 and len(a.split()) >= 15:
                    pairs.append({
                        "question": q,
                        "answer": a,
                        "analyst": question_speaker,
                        "executive": answer_speaker,
                        "ticker": record.get("ticker", ""),
                        "sector": record.get("sector", ""),
                        "title": record.get("title", ""),
                        "quarter": record.get("quarter", ""),
                        "year": record.get("year", ""),
                        "publish_date": record.get("publishOn", ""),
                        "transcript_id": record.get("id", ""),
                    })
            i = j
        else:
            i += 1

    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    all_pairs = []
    files = sorted(glob.glob(os.path.join(RAW_TRANSCRIPT_DIR, "*", "*.json")))
    logger.info("Found %d raw transcript files", len(files))

    with_qa, without_qa = 0, 0

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                continue
            record = json.loads(content)
        except Exception as exc:
            logger.warning("Skipping %s: %s", fpath, exc)
            continue

        # Try HTML parsing first (new API), then text (old import)
        html = record.get("content_html", "")
        if html and "structured-transcript" in html:
            pairs = extract_qa_from_html(record)
        else:
            pairs = extract_qa_from_text(record)

        all_pairs.extend(pairs)
        ticker = record.get("ticker", "?")
        tid = record.get("id", "?")
        if pairs:
            with_qa += 1
            logger.info("  %s/%s: %d Q&A pairs", ticker, tid, len(pairs))
        else:
            without_qa += 1
            logger.debug("  %s/%s: 0 Q&A pairs", ticker, tid)

    logger.info("=" * 60)
    logger.info("Transcripts with Q&A: %d, without: %d", with_qa, without_qa)
    logger.info("Total Q&A pairs: %d", len(all_pairs))

    from collections import Counter
    for ticker, count in sorted(Counter(p["ticker"] for p in all_pairs).items()):
        logger.info("  %s: %d pairs", ticker, count)

    os.makedirs(os.path.dirname(QA_PAIRS_PATH), exist_ok=True)
    with open(QA_PAIRS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
    logger.info("Saved to %s", QA_PAIRS_PATH)


if __name__ == "__main__":
    main()
