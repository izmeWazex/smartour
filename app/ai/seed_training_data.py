"""
Seed Training Data
------------------
The default labeled dataset lives in ``seed_training_data.csv`` (this folder)
instead of Python source, so it can be edited as plain data.

Each row: text,intent[,response]. Built-in intents need no response; custom
intents do. See ``csv_data.py`` for the exact format.
"""

from pathlib import Path

from app.ai.csv_data import read_examples_csv

SEED_CSV_PATH = Path(__file__).resolve().parent / "seed_training_data.csv"

SEED_TRAINING_EXAMPLES = read_examples_csv(SEED_CSV_PATH)
