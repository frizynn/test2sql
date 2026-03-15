"""MCTS tree data structures, pure helpers, and MCTSPipeline strategy.

This module provides:
- MCTSNode dataclass for the search tree
- Paper-faithful P(a) scoring: 1/2 * (min + mean) per MCTS-SQL (Yuan et al.)
- UCT computation, score parsing, node selection, backpropagation (pure)
- MCTSPipeline(PipelineStrategy): async pipeline with direct->verify->MCTS-Refiner flow
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field

from config import (
    logger,
    MCTS_MAX_ROLLOUTS,
    MCTS_CHILD_NODES,
    MCTS_EXPLORATION_CONSTANT,
)
from llm_client import call_model_raw
from prompts import (
    build_text2sql_prompt,
    build_mcts_verify_prompt,
    build_mcts_critique_prompt,
    build_mcts_refine_prompt,
    build_mcts_evaluate_prompt,
)
from sql_utils import extract_sql, validate_sql_explain


# ---------------------------------------------------------------------------
# Sanity check (moved from sql_helpers)
# ---------------------------------------------------------------------------

def _check_result_sanity(question: str, rows: list[list]) -> str | None:
    """Return error string if results look suspicious, else None."""
    q = question.lower()

    if not rows:
        signals = ["cuánto", "cuáles", "cuál", "top", "total", "mozo",
                    "producto", "qué", "cuántos", "cuántas", "promedio",
                    "vendedor", "quien", "quién"]
        if any(s in q for s in signals):
            return "empty_result: query returned no rows but question expects data"

    if len(rows) == 1 and len(rows[0]) == 1:
        val = rows[0][0]
        if val == 0 and any(s in q for s in ["cuánto", "total", "cuántas", "cuántos"]):
            return "zero_result: count/total is 0 but question expects non-zero"

    return None


# ---------------------------------------------------------------------------
# MCTS tree types and pure helpers
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """A single node in the MCTS search tree."""

    sql: str
    scores: list[float] = field(default_factory=list)
    visits: int = 0
    children: list[MCTSNode] = field(default_factory=list)
    parent: MCTSNode | None = field(default=None, repr=False)
    critique: str = ""
    error: str = ""
    is_valid: bool = False

    @property
    def score(self) -> float:
        """Backward-compat: return P(a) value."""
        return compute_p_value(self)


def compute_p_value(node: MCTSNode) -> float:
    """Paper's P(a) with child-aware backprop per Eq. 4 + Eq. 8.

    Leaf nodes: P(a) = 1/2 * (min(scores) + mean(scores))  [Eq. 4]
    Internal nodes: P'(a) = 1/2 * (P(a) + max(P(child)))   [Eq. 8]

    Returns 0.0 for nodes with no scores.
    """
    if not node.scores:
        return 0.0
    p_self = 0.5 * (min(node.scores) + sum(node.scores) / len(node.scores))
    if not node.children:
        return p_self
    best_child_p = max(compute_p_value(c) for c in node.children)
    return 0.5 * (p_self + best_child_p)


def compute_uct(node: MCTSNode, exploration_constant: float) -> float:
    """Compute the Upper Confidence Bound for Trees (UCT) value.

    UCT = P(a) + C * sqrt( (ln(N_parent) + 1) / (N(a) + eps) )

    Matches Eq. 9 of the paper.
    Returns ``float('inf')`` for unvisited nodes (visits == 0) so they
    are selected first.  If *node* has no parent, returns P(a) directly.
    """
    if node.visits == 0:
        return float("inf")

    p_val = compute_p_value(node)

    if node.parent is None:
        return p_val

    exploration = exploration_constant * math.sqrt(
        (math.log(node.parent.visits) + 1) / (node.visits + 1e-8)
    )
    return p_val + exploration


def parse_evaluate_score(raw_text: str) -> int:
    """Extract an integer score from messy LLM output and clamp to [-95, 95].

    Looks for the first integer (possibly negative) in *raw_text*.
    Returns 0 if no integer is found.
    """
    match = re.search(r"-?\d+", raw_text)
    if match is None:
        return 0
    value = int(match.group())
    return max(-95, min(95, value))


def select_node(root: MCTSNode, exploration_constant: float) -> MCTSNode:
    """Walk tree from *root*, always picking the child with highest UCT.

    Stops at a leaf node (no children).  Returns the selected node.
    """
    current = root
    while current.children:
        current = max(
            current.children,
            key=lambda n: compute_uct(n, exploration_constant),
        )
    return current


def backpropagate(node: MCTSNode, score: float) -> None:
    """Record score on *node* and bump visit counters up to root.

    Per the paper (Eq. 4 + Eq. 8):
    - Each node keeps its OWN scores (not descendants').
    - P(leaf) = 1/2 * (min + mean) of its own scores.
    - P'(parent) = 1/2 * (P(parent) + max P(children)) -- computed on the fly.

    Backprop only records the score on the leaf node and increments visit
    counts on all ancestors so UCT exploration tracking stays correct.
    """
    node.scores.append(score)
    current: MCTSNode | None = node
    while current is not None:
        current.visits += 1
        current = current.parent


# ---------------------------------------------------------------------------
# MCTSPipeline -- direct generation + MCTS-Refiner
# ---------------------------------------------------------------------------

# Import here to avoid circular import (PipelineStrategy is defined in __init__)
from pipelines import PipelineContext, PipelineResult, PipelineStrategy  # noqa: E402

FEW_SHOT_K = 5


class MCTSPipeline(PipelineStrategy):
    """Two-phase text-to-SQL pipeline: direct generation -> MCTS-Refiner.

    Fast path: generate SQL at temperature=0 with grammar, validate & execute.
    If valid, return immediately.  Otherwise activate the MCTS-Refiner loop
    (critique -> refine -> evaluate) for up to MCTS_MAX_ROLLOUTS iterations.
    """

    async def run(self, question: str, on_event=None) -> PipelineResult:
        trace: list[dict] = []
        meta: dict = {
            "mcts_activated": False,
            "rollouts_used": 0,
            "node_scores": [],
            "best_score": 0.0,
            "tree_depth": 0,
            "direct_sql": "",
            "direct_error": "",
            "verifier_result": "",
            "trace": trace,
        }

        async def _emit(event: dict) -> None:
            if on_event is not None:
                await on_event(event)

        await _emit({"type": "direct_generate_start"})
        sql, error, results = await self._direct_generate(question)
        meta["direct_sql"] = sql
        meta["direct_error"] = error

        trace.append({
            "step": "direct_generate",
            "status": "ok" if (results and not error) else "error",
            "sql": sql,
            "error": error or None,
        })

        await _emit({
            "type": "direct_generate_done",
            "status": "ok" if (results and not error) else "error",
            "sql": sql,
            "error": error or None,
        })

        if results is not None and not error:
            verifier_ok, verifier_raw = await self._verify_sql(question, sql)
            meta["verifier_result"] = "ok" if verifier_ok else "rejected"

            trace.append({
                "step": "verifier",
                "status": "ok" if verifier_ok else "rejected",
                "detail": verifier_raw,
            })

            await _emit({
                "type": "verifier",
                "status": "ok" if verifier_ok else "rejected",
                "detail": verifier_raw,
            })

            if verifier_ok:
                logger.info(json.dumps({
                    "event": "mcts_direct_ok",
                    "question": question[:120],
                    "sql": sql,
                    "verified": True,
                }))
                return PipelineResult(
                    sql=sql,
                    results=results,
                    attempts=1,
                    metadata=meta,
                )
            else:
                error = "verifier_rejected: SQL may not match user intent"
                logger.info(json.dumps({
                    "event": "mcts_verifier_rejected",
                    "question": question[:120],
                    "sql": sql,
                }))

        meta["mcts_activated"] = True
        return await self._mcts_refine(question, sql, error, meta, on_event)

    # -- LLM Verifier (paper Algorithm 1 step 3) --------------------------------

    async def _verify_sql(self, question: str, sql: str) -> tuple[bool, str]:
        """Call LLM verifier to check if SQL semantically matches user intent."""
        ctx = self.ctx
        prompt = build_mcts_verify_prompt(question, sql)
        mr = await call_model_raw(
            ctx.http_client,
            messages=[{"role": "user", "content": prompt}],
            url=ctx.nlg_url,
            timeout=ctx.model_timeout,
            temperature=0,
            max_tokens=128,
        )

        if mr["error"]:
            logger.info(json.dumps({
                "event": "mcts_verifier_error",
                "error": mr["error"],
            }))
            return True, f"error: {mr['error']}"

        raw = mr["content"].strip()
        is_ok = raw.upper().startswith("YES")
        logger.info(json.dumps({
            "event": "mcts_verifier_result",
            "question": question[:80],
            "sql": sql[:200],
            "result": "ok" if is_ok else "rejected",
            "raw": raw[:100],
        }))
        return is_ok, raw[:200]

    # -- Direct generation fast path -------------------------------------------

    async def _direct_generate(
        self, question: str,
    ) -> tuple[str, str, dict | None]:
        """Generate SQL directly, validate, execute, sanity-check.

        Returns (sql, error_or_empty, results_dict_or_none).
        """
        ctx = self.ctx

        few_shots = None
        if ctx.text2sql_retriever is not None:
            few_shots = ctx.text2sql_retriever.retrieve(question, k=FEW_SHOT_K)

        prompt = build_text2sql_prompt(question, few_shots=few_shots)
        messages = [{"role": "user", "content": prompt}]

        mr = await call_model_raw(
            ctx.http_client,
            messages=messages,
            url=ctx.text2sql_url,
            timeout=ctx.model_timeout,
            temperature=0,
            grammar=ctx.grammar or None,
            max_tokens=1024,
        )

        if mr["error"]:
            return "", mr["error"], None

        sql = extract_sql(mr["content"])
        if not sql:
            return "", "empty_model_output", None

        valid, err_msg = validate_sql_explain(sql, ctx.db_path)
        if not valid:
            return sql, err_msg, None

        try:
            conn = sqlite3.connect(ctx.db_path)
            cur = conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [list(r) for r in cur.fetchall()]
            conn.close()
        except sqlite3.Error as exc:
            return sql, str(exc), None

        sanity = _check_result_sanity(question, rows)
        if sanity:
            return sql, sanity, None

        return sql, "", {"columns": cols, "rows": rows, "row_count": len(rows)}

    # -- Single child expansion ------------------------------------------------

    async def _expand_one_child(
        self,
        question: str,
        node: MCTSNode,
        critique_text: str,
        child_idx: int,
    ) -> tuple[str | None, str, bool, dict | None, float]:
        """Generate one refined SQL child, validate, execute, evaluate.

        Returns (refined_sql, error, is_valid, results_dict, score).
        Temperature ramps with child_idx to encourage diversity across siblings.
        """
        ctx = self.ctx

        temp = 0.1 + child_idx * 0.15

        refine_prompt = build_mcts_refine_prompt(
            question, node.sql, critique_text, error=node.error,
        )
        refine_mr = await call_model_raw(
            ctx.http_client,
            messages=[{"role": "user", "content": refine_prompt}],
            url=ctx.text2sql_url,
            timeout=ctx.model_timeout,
            temperature=temp,
            grammar=ctx.grammar or None,
            max_tokens=1024,
        )

        if refine_mr["error"]:
            return None, refine_mr["error"], False, None, -95.0

        refined_sql = extract_sql(refine_mr["content"])
        if not refined_sql:
            return None, "empty_model_output", False, None, -95.0

        child_error = ""
        child_valid = False
        child_results: dict | None = None

        valid, err_msg = validate_sql_explain(refined_sql, ctx.db_path)
        if not valid:
            child_error = err_msg
        else:
            try:
                conn = sqlite3.connect(ctx.db_path)
                cur = conn.cursor()
                cur.execute(refined_sql)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = [list(r) for r in cur.fetchall()]
                conn.close()

                sanity = _check_result_sanity(question, rows)
                if sanity:
                    child_error = sanity
                else:
                    child_valid = True
                    child_results = {
                        "columns": cols,
                        "rows": rows,
                        "row_count": len(rows),
                    }
            except sqlite3.Error as exc:
                child_error = str(exc)

        eval_prompt = build_mcts_evaluate_prompt(
            question, refined_sql, error=child_error,
        )
        eval_mr = await call_model_raw(
            ctx.http_client,
            messages=[{"role": "user", "content": eval_prompt}],
            url=ctx.text2sql_url,
            timeout=ctx.model_timeout,
            temperature=0,
            max_tokens=64,
        )

        raw_score = float(
            parse_evaluate_score(eval_mr["content"]) if not eval_mr["error"] else 0
        )

        if child_valid:
            raw_score = min(95.0, raw_score + 20)
        elif child_error:
            raw_score = max(-95.0, raw_score - 20)

        return refined_sql, child_error, child_valid, child_results, raw_score

    # -- MCTS-Refiner ----------------------------------------------------------

    async def _mcts_refine(
        self, question: str, root_sql: str, root_error: str, meta: dict,
        on_event=None,
    ) -> PipelineResult:
        """Run MCTS critique->refine->evaluate loop."""
        ctx = self.ctx
        child_count = MCTS_CHILD_NODES

        async def _emit(event: dict) -> None:
            if on_event is not None:
                await on_event(event)

        root = MCTSNode(sql=root_sql, error=root_error)

        logger.info(json.dumps({
            "event": "mcts_refiner_start",
            "question": question[:120],
            "root_sql": root_sql,
            "root_error": root_error[:200],
            "child_nodes": child_count,
        }))

        trace = meta["trace"]

        await _emit({"type": "mcts_start"})

        for rollout_idx in range(1, MCTS_MAX_ROLLOUTS + 1):
            node = select_node(root, MCTS_EXPLORATION_CONSTANT)

            await _emit({"type": "rollout_start", "rollout": rollout_idx})

            # -- Critique (once per rollout) --
            critique_prompt = build_mcts_critique_prompt(
                question, node.sql, node.error,
            )
            critique_mr = await call_model_raw(
                ctx.http_client,
                messages=[{"role": "user", "content": critique_prompt}],
                url=ctx.text2sql_url,
                timeout=ctx.model_timeout,
                temperature=0,
                max_tokens=256,
            )

            if critique_mr["error"]:
                backpropagate(node, -95)
                meta["node_scores"].append(-95)
                trace.append({
                    "step": "rollout",
                    "rollout": rollout_idx,
                    "status": "error",
                    "error": critique_mr["error"],
                })
                logger.info(json.dumps({
                    "event": "mcts_rollout",
                    "rollout": rollout_idx,
                    "critique_error": critique_mr["error"],
                    "score": -95,
                }))
                meta["rollouts_used"] = rollout_idx
                continue

            critique_text = critique_mr["content"]

            await _emit({
                "type": "critique",
                "rollout": rollout_idx,
                "text": critique_text[:300],
            })

            # -- Generate child_count children (diverse refinements) --
            rollout_children = []
            for child_idx in range(child_count):
                refined_sql, child_error, child_valid, child_results, raw_score = (
                    await self._expand_one_child(
                        question, node, critique_text, child_idx,
                    )
                )

                child = MCTSNode(
                    sql=refined_sql or node.sql,
                    error=child_error,
                    is_valid=child_valid,
                    critique=critique_text,
                    parent=node,
                )
                node.children.append(child)
                backpropagate(child, raw_score)

                meta["node_scores"].append(raw_score)
                rollout_children.append({
                    "sql": (refined_sql or "")[:300],
                    "score": raw_score,
                    "valid": child_valid,
                    "error": child_error[:150] if child_error else None,
                })

                await _emit({
                    "type": "child",
                    "rollout": rollout_idx,
                    "child": child_idx,
                    "sql": (refined_sql or "")[:300],
                    "score": raw_score,
                    "valid": child_valid,
                    "error": child_error[:150] if child_error else None,
                })

                logger.info(json.dumps({
                    "event": "mcts_rollout",
                    "rollout": rollout_idx,
                    "child": child_idx,
                    "refined_sql": (refined_sql or "")[:200],
                    "score": raw_score,
                    "is_valid": child_valid,
                    "critique": critique_text[:100],
                }))

                if child_results is not None:
                    child._results = child_results  # type: ignore[attr-defined]

            trace.append({
                "step": "rollout",
                "rollout": rollout_idx,
                "status": "ok",
                "parent_sql": node.sql[:200],
                "critique": critique_text[:300],
                "children": rollout_children,
            })
            meta["rollouts_used"] = rollout_idx

        # -- Pick best valid node (by raw score, not P-value) --
        all_nodes = _collect_all_nodes(root)
        meta["tree_depth"] = _tree_depth(root)

        valid_nodes = [n for n in all_nodes if n.is_valid]

        if valid_nodes:
            best = max(valid_nodes, key=lambda n: max(n.scores) if n.scores else 0.0)
            meta["best_score"] = max(best.scores) if best.scores else 0.0
            trace.append({
                "step": "best_node",
                "status": "ok",
                "sql": best.sql[:300],
                "score": meta["best_score"],
                "valid_candidates": len(valid_nodes),
            })

            await _emit({
                "type": "best_node",
                "status": "ok",
                "sql": best.sql[:300],
                "score": meta["best_score"],
                "valid_candidates": len(valid_nodes),
            })

            best_results = getattr(best, "_results", None)
            if best_results is None:
                try:
                    conn = sqlite3.connect(ctx.db_path)
                    cur = conn.cursor()
                    cur.execute(best.sql)
                    cols = [d[0] for d in cur.description] if cur.description else []
                    rows = [list(r) for r in cur.fetchall()]
                    conn.close()
                    best_results = {"columns": cols, "rows": rows, "row_count": len(rows)}
                except sqlite3.Error:
                    best_results = None

            logger.info(json.dumps({
                "event": "mcts_best_node",
                "sql": best.sql,
                "score": best.score,
                "tree_depth": meta["tree_depth"],
            }))

            return PipelineResult(
                sql=best.sql,
                results=best_results,
                attempts=1 + meta["rollouts_used"],
                metadata=meta,
            )

        # No valid node
        trace.append({
            "step": "best_node",
            "status": "all_failed",
            "valid_candidates": 0,
        })

        await _emit({
            "type": "best_node",
            "status": "all_failed",
            "sql": "",
            "score": 0.0,
            "valid_candidates": 0,
        })

        best_any = max(
            all_nodes,
            key=lambda n: max(n.scores) if n.scores else -999.0,
        ) if all_nodes else root
        meta["best_score"] = max(best_any.scores) if best_any.scores else 0.0

        logger.info(json.dumps({
            "event": "mcts_best_node",
            "sql": best_any.sql,
            "score": best_any.score,
            "all_failed": True,
        }))

        return PipelineResult(
            sql=best_any.sql,
            attempts=1 + meta["rollouts_used"],
            error="mcts_all_nodes_failed",
            error_detail=(
                f"All {meta['rollouts_used']} MCTS rollouts produced invalid SQL. "
                f"Best score: {best_any.score}. Root error: {root_error[:200]}"
            ),
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Tree traversal helpers
# ---------------------------------------------------------------------------

def _collect_all_nodes(root: MCTSNode) -> list[MCTSNode]:
    """BFS to collect every node in the tree."""
    result: list[MCTSNode] = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        result.append(node)
        queue.extend(node.children)
    return result


def _tree_depth(root: MCTSNode) -> int:
    """Compute the depth of the tree."""
    if not root.children:
        return 0
    return 1 + max(_tree_depth(c) for c in root.children)
