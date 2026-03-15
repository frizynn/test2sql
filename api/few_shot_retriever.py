"""Few-shot example retriever using sentence embeddings and cosine similarity.

Loads a JSON pool of few-shot examples, pre-computes embeddings with
all-MiniLM-L6-v2, and retrieves top-k most similar examples for a question.

Boundary contract:
    __init__(pool_path, model_name) — loads and embeds the pool
    retrieve(question, k)          — returns top-k example dicts ranked by similarity
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean_pool(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-weighted mean pooling over token embeddings."""
    token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden)
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())  # (batch, seq_len, hidden)
    summed = torch.sum(token_embeddings * mask_expanded.float(), dim=1)           # (batch, hidden)
    counts = torch.clamp(mask_expanded.float().sum(dim=1), min=1e-9)              # (batch, hidden)
    return summed / counts


def _embed_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
) -> torch.Tensor:
    """Tokenize, encode, mean-pool, and L2-normalize a batch of texts.

    Returns:
        Tensor of shape (len(texts), hidden_dim) with unit-length rows.
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(**encoded)
    pooled = _mean_pool(output, encoded["attention_mask"])
    return F.normalize(pooled, p=2, dim=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FewShotRetriever:
    """Retrieve top-k few-shot examples by semantic similarity to a question.

    Args:
        pool_path: Path to a JSON file containing a list of example dicts.
                   Each dict must have a ``"question"`` field.
        model_name: HuggingFace model identifier. Defaults to
                    ``config.EMBEDDING_MODEL_NAME``.
    """

    def __init__(self, pool_path: str, model_name: Optional[str] = None):
        model_name = model_name or config.EMBEDDING_MODEL_NAME
        t0 = time.time()

        # Load pool -----------------------------------------------------------
        try:
            with open(pool_path, "r", encoding="utf-8") as f:
                self._pool = json.load(f)  # type: List[Dict[str, Any]]
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Failed to load few-shot pool from %s: %s" % (pool_path, exc)
            ) from exc

        if not isinstance(self._pool, list):
            raise ValueError("Few-shot pool must be a JSON array, got %s" % type(self._pool).__name__)

        # Load model ----------------------------------------------------------
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load embedding model '%s': %s" % (model_name, exc)
            ) from exc

        # Pre-compute embeddings ----------------------------------------------
        if self._pool:
            questions = [ex["question"] for ex in self._pool]
            self._pool_embeddings = _embed_texts(questions, self._tokenizer, self._model)
        else:
            # Empty pool — store a zero-row tensor to keep .shape consistent
            self._pool_embeddings = torch.empty(0, 384)
            config.logger.warning("Few-shot pool is empty: %s", pool_path)

        elapsed = time.time() - t0
        config.logger.info(
            "FewShotRetriever: loaded %d examples, embedded in %.2fs (%s)",
            len(self._pool),
            elapsed,
            model_name,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str, k: int = 10) -> List[Dict[str, Any]]:
        """Return top-*k* pool examples most similar to *question*.

        Returns a list of example dicts in descending similarity order.
        If *k* exceeds the pool size, all examples are returned.
        If the pool is empty, returns an empty list.
        """
        if not self._pool:
            return []

        k = min(k, len(self._pool))

        query_emb = _embed_texts([question], self._tokenizer, self._model)  # (1, dim)
        similarities = torch.mm(query_emb, self._pool_embeddings.T).squeeze(0)  # (pool_size,)
        top_indices = torch.topk(similarities, k).indices.tolist()

        return [self._pool[i] for i in top_indices]
