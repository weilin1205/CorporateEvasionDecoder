"""
03_llm_annotate.py
Label Q&A pairs using a local LLM (Qwen3 or similar).

Labels:
  0 = DIRECT    — executive directly addresses the question
  1 = EVASIVE   — executive deflects or avoids the question
  2 = JARGON    — excessive jargon obscures a lack of substance

Output: data/labeled_dataset.csv

Usage:
    python 03_llm_annotate.py
    python 03_llm_annotate.py --model /path/to/model
"""

import argparse
import json
import os
import re
import logging

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    LLM_MODEL_PATH, QA_PAIRS_PATH, LABELED_DATASET_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP = {"DIRECT": 0, "EVASIVE": 1, "JARGON": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

SYSTEM_PROMPT = (
    "Classify the executive's earnings-call response as exactly one of: "
    "DIRECT, EVASIVE, or JARGON. Output ONLY that single word, nothing else.\n\n"
    "DIRECT = specific numbers, facts, or clear explanations that address the question.\n"
    "EVASIVE = deflects, redirects, gives a non-answer, or avoids the question.\n"
    "JARGON = excessive buzzwords or vague corporate language hiding lack of substance."
)

USER_TEMPLATE = "Question: {question}\n\nResponse: {answer}\n\nLabel:"


def build_messages(question: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                question=question[:32768],
                answer=answer[:1024],
            ),
        },
    ]


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|.*?\|>", "", text)  # strip special tokens
    return text.strip()


def parse_label(raw: str) -> int:
    """Extract the classification label from the model output."""
    text = strip_thinking(raw).upper().strip()

    # Best case: the output is exactly one of the labels
    first_word = text.split()[0] if text.split() else ""
    first_word = re.sub(r"[^A-Z]", "", first_word)
    if first_word in LABEL_MAP:
        return LABEL_MAP[first_word]

    # Search for label keywords anywhere
    for label_name, label_id in LABEL_MAP.items():
        if label_name in text:
            return label_id
    if "EVAS" in text:
        return 1
    if "JARG" in text:
        return 2

    logger.warning("Unparseable output: %s", text[:100])
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM_MODEL_PATH)
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="Max new tokens (needs room for thinking tags)")
    ap.add_argument("--input", default=QA_PAIRS_PATH)
    ap.add_argument("--output", default=LABELED_DATASET_PATH)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    logger.info("Loaded %d Q&A pairs", len(qa_pairs))

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    labels: list[int] = []
    raw_outputs: list[str] = []

    # Process one at a time to avoid batched-padding issues
    for i, item in enumerate(tqdm(qa_pairs, desc="Annotating")):
        messages = build_messages(item["question"], item["answer"])

        # Try with enable_thinking=False for Qwen3
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

        label = parse_label(decoded)
        labels.append(label)
        raw_outputs.append(strip_thinking(decoded)[:200])

        # Log progress every 50 samples
        if (i + 1) % 50 == 0:
            parsed = sum(1 for l in labels if l != -1)
            logger.info("Progress: %d/%d processed, %d parsed (%.1f%%)",
                        i + 1, len(qa_pairs), parsed, 100 * parsed / (i + 1))

    # Build DataFrame
    df = pd.DataFrame(qa_pairs)
    df["label"] = labels
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df["llm_raw_output"] = raw_outputs

    # Report stats before dropping
    total = len(df)
    bad = (df["label"] == -1).sum()
    logger.info("Annotation complete: %d total, %d parsed, %d unparseable (%.1f%%)",
                total, total - bad, bad, 100 * bad / total)

    if bad > 0:
        logger.warning("Dropping %d unparseable rows", bad)
        df = df[df["label"] != -1].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved %d labeled samples to %s", len(df), args.output)
    logger.info("Label distribution:\n%s", df["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
