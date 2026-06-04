"""
Text feature preparation: prompt formatting for Gemma-2B LoRA fine-tuning.
"""
import pandas as pd


PROMPT_TEMPLATE = (
    "Classify the following fundraiser as FRAUD or LEGITIMATE.\n\n"
    "Title: {title}\n"
    "Description: {description}\n\n"
    "Answer (FRAUD or LEGITIMATE):"
)

LABEL_MAP = {0: "LEGITIMATE", 1: "FRAUD"}
REVERSE_LABEL_MAP = {"LEGITIMATE": 0, "FRAUD": 1}


def format_prompt(title: str, description: str) -> str:
    return PROMPT_TEMPLATE.format(
        title=title.strip() if title else "N/A",
        description=(description.strip()[:1000] if description else "N/A"),
    )


def build_text_dataset(df: pd.DataFrame, label_col: str = "label") -> list[dict]:
    """
    Build a list of {input_text, label} dicts for HuggingFace Dataset.
    Uses description_clean if available, falls back to description.
    """
    desc_col = "description_clean" if "description_clean" in df.columns else "description"
    records = []
    for _, row in df.iterrows():
        records.append({
            "input_text": format_prompt(
                str(row.get("title", "")),
                str(row.get(desc_col, ""))
            ),
            "label": int(row[label_col]),
            "fund_id": int(row["fund_id"]),
        })
    return records


def decode_prediction(logits_or_label: int) -> str:
    return LABEL_MAP.get(logits_or_label, "UNKNOWN")
