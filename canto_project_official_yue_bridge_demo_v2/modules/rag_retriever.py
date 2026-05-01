"""
RAG retriever for Cantopop lyrics corpus.

Builds a FAISS vector index from cantopop_corpus_final_583_yue.csv and
provides retrieve() to fetch the top-k most similar lyrics for a query.
Designed to be imported lazily so the app only loads the heavy dependencies
when RAG is actually requested.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# ── Lazy singletons ─────────────────────────────────────────────────────────
_embedder = None
_index = None
_corpus: List[str] = []

CSV_PATH: Path | None = None  # set by init()


def init(csv_path: str | Path) -> None:
    """Load corpus and build FAISS index. Call once; subsequent calls are no-ops."""
    global _embedder, _index, _corpus, CSV_PATH

    if _index is not None:
        return  # already initialised

    from sentence_transformers import SentenceTransformer
    import faiss

    CSV_PATH = Path(csv_path)
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"RAG corpus not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Lyrics_YuE"])
    df = df[df["Lyrics_YuE"].str.strip().str.len() > 50].reset_index(drop=True)
    _corpus = df["Lyrics_YuE"].tolist()

    _embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = _embedder.encode(
        _corpus, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(embeddings)


def retrieve(query: str, top_k: int = 3) -> List[Dict]:
    """Return top_k most similar lyrics to query. init() must have been called."""
    if _index is None or _embedder is None:
        raise RuntimeError("RAG retriever not initialised. Call rag_retriever.init(csv_path) first.")

    q_emb = _embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({"score": float(score), "lyrics": _corpus[idx]})
    return results


def build_few_shot_block(query: str, top_k: int = 3, max_chars_per_lyric: int = 500) -> str:
    """Return a formatted few-shot block to inject into the generation prompt."""
    examples = retrieve(query, top_k=top_k)
    lines = ["以下是一些優質粵語歌詞的例子，供你參考格式、押韻與風格：\n"]
    for i, ex in enumerate(examples, 1):
        preview = ex["lyrics"].strip()[:max_chars_per_lyric]
        lines.append(f"--- 參考歌詞 {i} ---\n{preview}\n")
    return "\n".join(lines)
