"""
CSV helpers for training data.
Format: text,intent[,response] (header optional, UTF-8).
"""

import csv
import io
from pathlib import Path
from typing import List, Optional, Union

# backend/data/ — user datasets (gitignored)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATASET_CSV_PATH = DATA_DIR / "training_data.csv"

EXPECTED_HEADER = ("text", "intent")


def _parse_csv_rows(source) -> List[List[str]]:
    reader = csv.reader(source)
    rows = []
    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            continue
        rows.append(row)
    return rows


def read_examples_csv(source: Union[str, Path, object], encoding: str = "utf-8-sig") -> List[dict]:
    """
    Read labeled examples from a CSV file path or file-like object.

    Returns a list of ``{"text", "intent", "response"}`` dicts (response may be
    None). Raises ValueError on malformed rows.
    """
    if isinstance(source, (str, Path)):
        with open(source, newline="", encoding=encoding) as f:
            rows = _parse_csv_rows(f)
    else:
        # file-like object (e.g. UploadFile.file, io.StringIO)
        if hasattr(source, "read"):
            text = source.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8-sig")
            rows = _parse_csv_rows(io.StringIO(text))
        else:
            raise ValueError("source must be a path or a readable file-like object")

    if not rows:
        raise ValueError("CSV file is empty.")

    # Optional header row
    first = [cell.strip().lower() for cell in rows[0]]
    if first[:2] == list(EXPECTED_HEADER):
        rows = rows[1:]

    examples: List[dict] = []
    for i, row in enumerate(rows, start=1):
        text = row[0].strip()
        intent = row[1].strip() if len(row) > 1 else ""
        response = row[2].strip() if len(row) > 2 else ""
        if not text or not intent:
            raise ValueError(
                f"CSV row {i} is missing 'text' or 'intent'. "
                f"Expected columns: text,intent[,response]."
            )
        examples.append({
            "text": text,
            "intent": intent,
            "response": response or None,
        })
    return examples


def write_examples_csv(examples: List[dict], path: Path = DATASET_CSV_PATH) -> Path:
    """Write labeled examples to a CSV file (with header), UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent", "response"])
        for ex in examples:
            writer.writerow([
                ex["text"],
                ex["intent"],
                ex.get("response") or "",
            ])
    return path
