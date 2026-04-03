"""
06_augment_data.py
Generate augmented training data by rewriting Q&A pairs using an LLM.

Strategy:
  - Rewrite EVASIVE answers → DIRECT style (labeled as DIRECT)
  - Rewrite DIRECT answers → EVASIVE style (labeled as EVASIVE)
  - This tests whether models learn the *structure* of evasion

Output: data/augmented_dataset.csv

Usage:
    python 06_augment_data.py
    python 06_augment_data.py --model /path/to/model
"""

import argparse
import logging
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    LLM_MODEL_PATH,
    LABELED_DATASET_PATH,
    AUGMENTED_DATASET_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REWRITE_EVASIVE_TO_DIRECT = """\
You are an expert corporate communication editor. Below is a question from \
a financial analyst and an evasive executive response. Rewrite ONLY the \
executive's response to be DIRECT: address the question head-on with \
specific, concrete information. Keep the same general topic and approximate \
length. Output ONLY the rewritten response, nothing else.

Question: {question}

Evasive Response: {answer}

Direct Rewrite:"""

REWRITE_DIRECT_TO_EVASIVE = """\
You are an expert corporate communication editor. Below is a question from \
a financial analyst and a direct executive response. Rewrite ONLY the \
executive's response to be EVASIVE: deflect the question, use vague language, \
redirect to a different topic, or give a non-answer. Keep approximate length. \
Output ONLY the rewritten response, nothing else.

Question: {question}

Direct Response: {answer}

Evasive Rewrite:"""


def strip_thinking(text: str) -> str:
    """Remove Qwen3 thinking blocks and special tokens from generated text."""
    text = re.sub(
        r"<redacted_thinking>.*?</redacted_thinking>", "", text, flags=re.DOTALL
    )
    text = re.sub(r"<redacted_thinking>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|.*?\|>", "", text)
    return text.strip()


def is_valid_rewrite(text: str) -> bool:
    """Reject generations that leaked reasoning, prompts, or chat scaffolding."""
    if len(text.split()) < 10:
        return False
    low = text.lower()
    if "redacted_thinking" in low:
        return False
    if "thinking process:" in low[:800]:
        return False
    if "you are an expert corporate communication editor" in low:
        return False
    if re.match(r"^\s*(user|assistant)\b", text, re.IGNORECASE):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=LLM_MODEL_PATH)
    parser.add_argument("--max-samples", type=int, default=300,
                        help="Max samples per direction to augment")
    args = parser.parse_args()

    df = pd.read_csv(LABELED_DATASET_PATH)
    logger.info("Loaded %d labeled samples", len(df))

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    augmented_rows = []
    skipped = {"evasive_to_direct": 0, "direct_to_evasive": 0}

    def build_prompt(user_msg: str) -> str:
        msgs = [{"role": "user", "content": user_msg}]
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    def generate_rewrite(prompt_text: str) -> str:
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.5,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        return strip_thinking(raw)

    # Direction 1: EVASIVE → DIRECT
    evasive_df = df[df["label"] == 1].head(args.max_samples)
    logger.info("Rewriting %d EVASIVE → DIRECT", len(evasive_df))
    for _, row in tqdm(evasive_df.iterrows(), total=len(evasive_df), desc="Evasive→Direct"):
        user_msg = REWRITE_EVASIVE_TO_DIRECT.format(
            question=row["question"][:1000], answer=row["answer"][:1500],
        )
        new_text = generate_rewrite(build_prompt(user_msg))
        if is_valid_rewrite(new_text):
            augmented_rows.append({
                "question": row["question"],
                "answer": new_text,
                "label": 0,
                "label_name": "DIRECT",
                "augmentation": "evasive_to_direct",
                "original_label": 1,
                "ticker": row.get("ticker", ""),
                "sector": row.get("sector", ""),
            })
        else:
            skipped["evasive_to_direct"] += 1

    # Direction 2: DIRECT → EVASIVE
    direct_df = df[df["label"] == 0].head(args.max_samples)
    logger.info("Rewriting %d DIRECT → EVASIVE", len(direct_df))
    for _, row in tqdm(direct_df.iterrows(), total=len(direct_df), desc="Direct→Evasive"):
        user_msg = REWRITE_DIRECT_TO_EVASIVE.format(
            question=row["question"][:1000], answer=row["answer"][:1500],
        )
        new_text = generate_rewrite(build_prompt(user_msg))
        if is_valid_rewrite(new_text):
            augmented_rows.append({
                "question": row["question"],
                "answer": new_text,
                "label": 1,
                "label_name": "EVASIVE",
                "augmentation": "direct_to_evasive",
                "original_label": 0,
                "ticker": row.get("ticker", ""),
                "sector": row.get("sector", ""),
            })
        else:
            skipped["direct_to_evasive"] += 1

    df_aug = pd.DataFrame(augmented_rows)
    df_aug.to_csv(AUGMENTED_DATASET_PATH, index=False)
    logger.info(
        "Skipped (invalid/leaked output): evasive→direct=%d, direct→evasive=%d",
        skipped["evasive_to_direct"],
        skipped["direct_to_evasive"],
    )
    logger.info("Saved %d augmented samples to %s", len(df_aug), AUGMENTED_DATASET_PATH)
    logger.info("Augmented label distribution:\n%s",
                df_aug["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
