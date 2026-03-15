"""Prompt building functions for text-to-SQL and NLG pipelines."""

from __future__ import annotations

from config import _PROMPT_TEMPLATE, _SCHEMA_TEXT, _NLG_TEMPLATE


# ---------------------------------------------------------------------------
# Core prompt builders
# ---------------------------------------------------------------------------

def build_text2sql_prompt(
    question: str,
    template=None,
    schema=None,
    few_shots: list[dict] | None = None,
) -> str:
    """Build a text-to-SQL prompt by replacing {schema} and {question} placeholders.

    When *few_shots* is a non-None list, the static examples between the
    ``=== EXAMPLES ===`` and ``=== QUESTION ===`` markers are replaced with
    dynamically formatted examples.  When *few_shots* is ``None`` (default),
    the existing static template is used as-is for backward compatibility.
    """
    if template is None:
        template = _PROMPT_TEMPLATE
    if schema is None:
        schema = _SCHEMA_TEXT

    prompt = template

    # Dynamic few-shot injection
    if few_shots is not None:
        examples_marker = "=== EXAMPLES ==="
        question_marker = "=== QUESTION ==="
        start = prompt.find(examples_marker)
        end = prompt.find(question_marker)
        if start != -1 and end != -1:
            # Keep the markers, replace only the content between them
            dynamic_block = "\n".join(
                f"\nQuestion: {ex['question']}\nSQL: {ex['sql']}"
                for ex in few_shots
            )
            # Add trailing newline so the QUESTION marker stays on its own line
            if dynamic_block:
                dynamic_block += "\n"
            prompt = (
                prompt[: start + len(examples_marker)]
                + "\n"
                + dynamic_block
                + "\n"
                + prompt[end:]
            )

    return prompt.replace("{schema}", schema).replace("{question}", question)


def format_results_for_prompt(rows, max_rows: int = 20) -> str:
    """Convert a list-of-lists into a tuple-list string for NLG prompt injection.

    Returns ``"[]"`` for empty/falsy *rows*. Truncates to *max_rows*.
    """
    if not rows:
        return "[]"
    return str([tuple(r) for r in rows[:max_rows]])


# ---------------------------------------------------------------------------
# MCTS pipeline — Verifier / Critique / Refine / Evaluate prompts
# ---------------------------------------------------------------------------

_MCTS_SCHEMA_COMPACT = (
    "TABLE sales (date TEXT 'MM/DD/YYYY', week_day TEXT 'Monday-Sunday', "
    "hour TEXT 'HH:MM', ticket_number TEXT, waiter TEXT, product_name TEXT, "
    "quantity REAL, unitary_price REAL, total REAL)"
)

# Few-shot examples for MCTS sub-agents (critique, refine, evaluate, verify)
_MCTS_FEW_SHOTS = [
    {
        "question": "¿Cuántos productos diferentes hay?",
        "sql": "SELECT COUNT(DISTINCT product_name) FROM sales;",
    },
    {
        "question": "¿Cuál es el producto más vendido?",
        "sql": "SELECT product_name, SUM(quantity) AS total_qty FROM sales GROUP BY product_name ORDER BY total_qty DESC LIMIT 1;",
    },
    {
        "question": "¿Cuánto vendió el mozo Juan en total?",
        "sql": "SELECT SUM(total) FROM sales WHERE waiter = 'Juan';",
    },
]

_MCTS_FEW_SHOT_BLOCK = "\n".join(
    f"Q: {ex['question']}\nSQL: {ex['sql']}" for ex in _MCTS_FEW_SHOTS
)


def build_mcts_verify_prompt(
    question: str,
    sql: str,
    schema: str | None = None,
) -> str:
    """Build a verifier prompt: check if SQL semantically matches user intent.

    Per paper Section 3.3 / Algorithm 1 step 3: the LLM-based verifier checks
    whether the SQL correctly answers the question (beyond just executing).
    Returns YES if correct, NO + explanation if not.
    """
    schema_text = schema or _MCTS_SCHEMA_COMPACT
    return (
        "You are a SQL verification expert.\n"
        f"Schema: {schema_text}\n\n"
        "Examples:\n"
        f"{_MCTS_FEW_SHOT_BLOCK}\n\n"
        "Check whether the SQL query correctly answers the user question.\n"
        "Verify: correct tables, columns, filters, aggregations, joins, "
        "groupings, and ordering.\n"
        "If the SQL is correct, answer 'YES'.\n"
        "If the SQL is incorrect, answer 'NO' and briefly explain the error.\n\n"
        f"User question: {question}\n\n"
        f"SQL:\n{sql}\n\n"
        "Answer:"
    )


def build_mcts_critique_prompt(
    question: str,
    sql: str,
    error: str,
    schema: str | None = None,
) -> str:
    """Build a critique prompt: identify what is wrong with a failing SQL query.

    Instructions in English (Qwen2.5-Coder reasons better in English about SQL).
    Question stays in the user's language.  Output: natural-language critique.
    """
    schema_text = schema or _MCTS_SCHEMA_COMPACT
    error_trunc = error[:200] if error else "unknown error"
    return (
        "You are a SQL debugging expert.\n"
        f"Schema: {schema_text}\n\n"
        "Reference examples:\n"
        f"{_MCTS_FEW_SHOT_BLOCK}\n\n"
        f"User question: {question}\n\n"
        f"Failing SQL:\n{sql}\n\n"
        f"Error: {error_trunc}\n\n"
        "Identify concisely what is wrong with the SQL and how to fix it. "
        "Focus on table names, column names, syntax, and logic errors."
    )


def build_mcts_refine_prompt(
    question: str,
    sql: str,
    critique: str,
    schema: str | None = None,
    error: str | None = None,
) -> str:
    """Build a refine prompt: produce corrected SQL given a critique.

    Instructions in English.  Question stays in the user's language.
    Truncates critique to 200 chars.  Output: a single SQL SELECT statement.
    Per paper Eq. 6: refiner receives (sql, critique, error, schema).
    """
    schema_text = schema or _MCTS_SCHEMA_COMPACT
    critique_trunc = critique[:200] if critique else "no critique provided"
    error_trunc = error[:200] if error else ""
    error_block = f"Execution error: {error_trunc}\n\n" if error_trunc else ""
    return (
        "You are a SQL expert. Write a corrected SQL query.\n"
        f"Schema: {schema_text}\n\n"
        "Reference examples:\n"
        f"{_MCTS_FEW_SHOT_BLOCK}\n\n"
        f"User question: {question}\n\n"
        f"Previous (broken) SQL:\n{sql}\n\n"
        f"{error_block}"
        f"Critique: {critique_trunc}\n\n"
        "Write ONLY the corrected SQL SELECT statement. No explanation."
    )


def build_mcts_evaluate_prompt(
    question: str,
    sql: str,
    schema: str | None = None,
    error: str | None = None,
) -> str:
    """Build an evaluate prompt: score SQL quality from -95 to 95.

    Instructions in English.  Question stays in the user's language.
    Per paper Eq. 7: evaluator receives (refined SQL, error, schema).
    Output: a single integer score.
    """
    schema_text = schema or _MCTS_SCHEMA_COMPACT
    error_trunc = error[:200] if error else ""
    error_block = f"Execution result: {error_trunc}\n\n" if error_trunc else "Execution result: OK (no errors)\n\n"
    return (
        "You are a SQL quality evaluator.\n"
        f"Schema: {schema_text}\n\n"
        "Reference examples:\n"
        f"{_MCTS_FEW_SHOT_BLOCK}\n\n"
        "Rate how well the SQL query answers the user question.\n"
        "Score from -95 (completely wrong) to 95 (perfectly correct).\n\n"
        f"User question: {question}\n\n"
        f"SQL:\n{sql}\n\n"
        f"{error_block}"
        "Respond with ONLY an integer score, nothing else."
    )


def build_nlg_prompt(question: str, sql: str, results_str: str, template=None) -> str:
    """Build an NLG prompt by replacing ``{question}``, ``{sql}``, and ``{results}`` placeholders."""
    if template is None:
        template = _NLG_TEMPLATE
    return template.replace("{question}", question).replace("{sql}", sql).replace("{results}", results_str)
