from enum import StrEnum
from pathlib import Path

PROMPT_DIR = Path(__file__).parent


class Prompt(StrEnum):
    QA = "qa.txt"
    EVAL = "eval.txt"


def load_prompt(name: Prompt) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8")
