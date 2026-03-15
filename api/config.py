"""Configuration: environment variables, constants, paths, logging, and file-loading helpers."""

import logging
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.environ.get("CSV_PATH", os.path.join(SCRIPT_DIR, "..", "data.csv"))
DB_PATH = os.environ.get("DB_PATH", os.path.join(SCRIPT_DIR, "sales.db"))
SCHEMA_PATH = os.environ.get("SCHEMA_PATH", os.path.join(SCRIPT_DIR, "..", "prompts", "schema.sql"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

EXPECTED_ROW_COUNT = 24212

# Embedding-based few-shot retrieval
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FEW_SHOT_DIR = os.path.join(SCRIPT_DIR, "..", "prompts", "few_shots")

# Text-to-SQL pipeline configuration
PROMPT_TEMPLATE_PATH = os.environ.get(
    "PROMPT_TEMPLATE_PATH",
    os.path.join(SCRIPT_DIR, "..", "prompts", "text2sql.txt"),
)
TEXT2SQL_URL = os.environ.get(
    "TEXT2SQL_URL", "http://localhost:8081/v1/chat/completions"
)
GRAMMAR_PATH = os.environ.get(
    "GRAMMAR_PATH",
    os.path.join(SCRIPT_DIR, "..", "grammars", "sql_select.gbnf"),
)
MAX_ATTEMPTS = 3
MODEL_TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "120"))

# NLG (answer generation) configuration
NLG_URL = os.environ.get(
    "NLG_URL", "http://localhost:8082/v1/chat/completions"
)
NLG_PROMPT_TEMPLATE_PATH = os.environ.get(
    "NLG_PROMPT_TEMPLATE_PATH",
    os.path.join(SCRIPT_DIR, "..", "prompts", "nlg.txt"),
)

# MCTS pipeline configuration
MCTS_MAX_ROLLOUTS = int(os.environ.get("MCTS_MAX_ROLLOUTS", "5"))
MCTS_CHILD_NODES = int(os.environ.get("MCTS_CHILD_NODES", "2"))
MCTS_EXPLORATION_CONSTANT = float(os.environ.get("MCTS_EXPLORATION_CONSTANT", "1.414"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# File-loading helpers
# ---------------------------------------------------------------------------
def _load_file_safe(path: str, label: str) -> str:
    """Load a text file, returning empty string if not found."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("%s not found: %s", label, path)
        return ""


# Module-level artifact loading (available for component tests at import time)
_PROMPT_TEMPLATE = _load_file_safe(PROMPT_TEMPLATE_PATH, "Prompt template")
_SCHEMA_TEXT = _load_file_safe(SCHEMA_PATH, "Schema")
_GRAMMAR_TEXT = _load_file_safe(GRAMMAR_PATH, "GBNF grammar")
_NLG_TEMPLATE = _load_file_safe(NLG_PROMPT_TEMPLATE_PATH, "NLG prompt template")
