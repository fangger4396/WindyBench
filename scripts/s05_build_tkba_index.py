#!/usr/bin/env python3
"""
Build and query an offline RAG knowledge base from QA JSONL.

Acronym mapping:
- TKBA: Turbine Knowledge Base Archive (offline indexed store)
- TKRA: Turbine Knowledge Retrieval Augmentation (runtime retrieval usage)

Paper-aligned features:
1) Structural chunking: parent-doc -> child chunks with overlap
2) Bi-encoder vector mapping
3) HNSW indexing (FAISS)
4) Subsystem/component metadata filtering
5) Parent-child context restoration
6) Optional cross-encoder reranking

Example build:
python s05_build_tkba_index.py build \
  --qa-jsonl qa_dataset_debug.jsonl \
  --out-dir rag_kb \
  --bi-encoder sentence-transformers/all-MiniLM-L6-v2

Example query:
python s05_build_tkba_index.py query \
  --kb-dir rag_kb \
  --query "Pitch actuator pressure drop during high wind braking" \
  --component-id "Pitch System / Hydraulics" \
  --top-k 5 \
  --use-rerank
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "faiss is required. Install with: pip install faiss-cpu"
    ) from e

SentenceTransformer = None
CrossEncoder = None


def _lazy_import_sentence_transformers() -> None:
    global SentenceTransformer, CrossEncoder
    if SentenceTransformer is not None and CrossEncoder is not None:
        return
    try:
        from sentence_transformers import CrossEncoder as _CrossEncoder, SentenceTransformer as _SentenceTransformer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required for HF embedding/reranker. "
            "Install with: pip install sentence-transformers"
        ) from e
    SentenceTransformer = _SentenceTransformer
    CrossEncoder = _CrossEncoder


class OllamaEmbedder:
    def __init__(self, model: str, ollama_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.endpoint = ollama_url.rstrip("/") + "/api/embeddings"

    def _embed_one(self, text: str, timeout_sec: int = 120) -> List[float]:
        payload = {"model": self.model, "prompt": text}
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        emb = obj.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(f"Invalid embedding response from Ollama: {obj}")
        return [float(x) for x in emb]

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        vecs: List[List[float]] = []
        total = len(texts)
        for i, t in enumerate(texts, start=1):
            vecs.append(self._embed_one(t))
            if show_progress_bar and (i % 50 == 0 or i == total):
                print(f"embedding {i}/{total}")
        arr = np.array(vecs, dtype="float32")
        if normalize_embeddings:
            arr = l2_normalize(arr)
        return arr if convert_to_numpy else arr


@dataclass
class ParentDoc:
    parent_id: str
    component_id: str
    text: str
    metadata: Dict[str, Any]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def detokenize_words(words: List[str]) -> str:
    return " ".join(words)


def build_parent_text(row: Dict[str, Any]) -> Tuple[str, str]:
    manifest = row.get("manifest", {})
    evidence = row.get("evidence", [])
    component_id = str(manifest.get("component_id", "UnknownComponent"))

    lines = [
        f"SampleID: {row.get('sample_id', '')}",
        f"TurbineID: {row.get('turbine_id', '')}",
        f"AlarmID: {row.get('alarm_id', '')}",
        f"AlarmDesc: {row.get('alarm_desc', '')}",
        f"ComponentID: {component_id}",
        f"EventStart: {row.get('event_start', '')}",
        f"EventEnd: {row.get('event_end', '')}",
        "SymptomManifest:",
        f"- Severity: {manifest.get('severity_level', '')}",
        f"- MeanAbsZPrimary: {manifest.get('mean_abs_z_primary', '')}",
        f"- Trend: {manifest.get('trend_description', '')}",
        f"- OperatingState: {manifest.get('operating_state', '')}",
        "Question:",
        str(row.get("question", "")),
        "Answer:",
        str(row.get("answer", "")),
        "Evidence:",
    ]
    for e in evidence[:5]:
        lines.append(
            f"- Sensor={e.get('sensor_id','')}, mean_abs_z={e.get('mean_abs_z','')}, "
            f"trend={e.get('trend','')}, min={e.get('event_min','')}, max={e.get('event_max','')}"
        )
    return "\n".join(lines), component_id


def chunk_text(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    words = tokenize_words(text)
    if not words:
        return []
    chunks: List[str] = []
    step = max(1, chunk_tokens - overlap_tokens)
    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_tokens]
        if not chunk:
            continue
        chunks.append(detokenize_words(chunk))
        if i + chunk_tokens >= len(words):
            break
    return chunks


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def build_kb(
    qa_jsonl: Path,
    out_dir: Path,
    bi_encoder_name: str,
    embed_backend: str,
    ollama_embed_model: str,
    ollama_url: str,
    parent_tokens: int,
    child_tokens: int,
    child_overlap: int,
    hnsw_m: int,
    ef_construction: int,
    batch_size: int,
) -> None:
    rows = read_jsonl(qa_jsonl)
    if not rows:
        raise ValueError(f"No rows found: {qa_jsonl}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if embed_backend == "hf":
        _lazy_import_sentence_transformers()
        model = SentenceTransformer(bi_encoder_name)
    elif embed_backend == "ollama":
        model = OllamaEmbedder(model=ollama_embed_model, ollama_url=ollama_url)
    else:
        raise ValueError(f"Unknown embed backend: {embed_backend}")

    parent_docs: List[ParentDoc] = []
    child_texts: List[str] = []
    child_meta: List[Dict[str, Any]] = []

    for row in rows:
        sample_id = str(row.get("sample_id", f"sample_{len(parent_docs)}"))
        parent_text, component_id = build_parent_text(row)

        # Parent truncation with token budget to keep section coherence.
        parent_words = tokenize_words(parent_text)[:parent_tokens]
        parent_text = detokenize_words(parent_words)
        parent_id = f"parent::{sample_id}"
        parent_docs.append(
            ParentDoc(
                parent_id=parent_id,
                component_id=component_id,
                text=parent_text,
                metadata={
                    "sample_id": sample_id,
                    "alarm_id": row.get("alarm_id"),
                    "alarm_desc": row.get("alarm_desc"),
                    "component_id": component_id,
                },
            )
        )

        for idx, c in enumerate(chunk_text(parent_text, child_tokens, child_overlap)):
            child_id = len(child_texts)
            child_texts.append(c)
            child_meta.append(
                {
                    "child_id": child_id,
                    "child_idx_in_parent": idx,
                    "parent_id": parent_id,
                    "component_id": component_id,
                    "alarm_id": row.get("alarm_id"),
                }
            )

    emb = model.encode(
        child_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    emb = l2_normalize(emb)
    dim = emb.shape[1]

    index = faiss.IndexHNSWFlat(dim, hnsw_m)
    index.hnsw.efConstruction = ef_construction
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.add(emb)

    # Save artifacts
    faiss.write_index(index, str(out_dir / "child_hnsw.index"))
    np.save(out_dir / "child_embeddings.npy", emb)
    (out_dir / "child_texts.jsonl").write_text(
        "\n".join(json.dumps({"child_id": i, "text": t}, ensure_ascii=False) for i, t in enumerate(child_texts)),
        encoding="utf-8",
    )
    (out_dir / "child_meta.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in child_meta), encoding="utf-8"
    )
    (out_dir / "parent_docs.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "parent_id": p.parent_id,
                    "component_id": p.component_id,
                    "text": p.text,
                    "metadata": p.metadata,
                },
                ensure_ascii=False,
            )
            for p in parent_docs
        ),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "embed_backend": embed_backend,
                "bi_encoder": bi_encoder_name,
                "ollama_embed_model": ollama_embed_model if embed_backend == "ollama" else None,
                "ollama_url": ollama_url if embed_backend == "ollama" else None,
                "dim": dim,
                "num_parents": len(parent_docs),
                "num_children": len(child_texts),
                "chunking": {
                    "parent_tokens": parent_tokens,
                    "child_tokens": child_tokens,
                    "child_overlap": child_overlap,
                },
                "hnsw": {"M": hnsw_m, "ef_construction": ef_construction},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"KB built: {out_dir}")
    print(f"parents={len(parent_docs)}, children={len(child_texts)}, dim={dim}")


def load_kb(kb_dir: Path) -> Dict[str, Any]:
    cfg = json.loads((kb_dir / "config.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(kb_dir / "child_hnsw.index"))
    emb = np.load(kb_dir / "child_embeddings.npy")

    child_texts = {}
    for line in (kb_dir / "child_texts.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        child_texts[int(o["child_id"])] = o["text"]

    child_meta = {}
    for line in (kb_dir / "child_meta.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        child_meta[int(o["child_id"])] = o

    parents = {}
    for line in (kb_dir / "parent_docs.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        parents[o["parent_id"]] = o

    return {
        "config": cfg,
        "index": index,
        "emb": emb,
        "child_texts": child_texts,
        "child_meta": child_meta,
        "parents": parents,
    }


def retrieve(
    kb: Dict[str, Any],
    bi_encoder: Any,
    query: str,
    component_id: Optional[str],
    top_k: int,
    candidate_k: int,
    use_rerank: bool,
    reranker: Optional[CrossEncoder],
    alarm_id: Optional[str] = None,
    alarm_filter: str = "off",
) -> Dict[str, Any]:
    if alarm_filter not in {"off", "soft", "hard"}:
        raise ValueError(f"Invalid alarm_filter: {alarm_filter}")

    q = bi_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    q = l2_normalize(q)

    child_ids: List[int]
    child_scores: List[float]

    # Metadata filtering before similarity computation.
    filtered = list(kb["child_meta"].keys())
    if component_id:
        filtered = [
            cid
            for cid in filtered
            if str(kb["child_meta"][cid].get("component_id", "")).strip() == str(component_id).strip()
        ]
    if alarm_filter == "hard" and alarm_id is not None:
        wanted_alarm = str(alarm_id).strip()
        filtered_hard = [
            cid for cid in filtered if str(kb["child_meta"][cid].get("alarm_id", "")).strip() == wanted_alarm
        ]
        if filtered_hard:
            filtered = filtered_hard
    if filtered:
        mat = kb["emb"][filtered]  # [N, D]
        sims = mat @ q[0]  # cosine due to normalization
        order = np.argsort(-sims)[:candidate_k]
        child_ids = [filtered[int(i)] for i in order]
        child_scores = [float(sims[int(i)]) for i in order]
    else:
        if component_id:
            return {"query": query, "component_id": component_id, "children": [], "parents": []}
        scores, ids = kb["index"].search(q, candidate_k)
        child_ids = [int(i) for i in ids[0] if int(i) >= 0]
        child_scores = [float(s) for s in scores[0][: len(child_ids)]]

    candidates = []
    for cid, sc in zip(child_ids, child_scores):
        candidates.append(
            {
                "child_id": cid,
                "score": sc,
                "text": kb["child_texts"][cid],
                "meta": kb["child_meta"][cid],
            }
        )

    if use_rerank and reranker is not None and candidates:
        pairs = [(query, c["text"]) for c in candidates]
        rr = reranker.predict(pairs)
        for c, r in zip(candidates, rr):
            c["rerank_score"] = float(r)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    else:
        candidates.sort(key=lambda x: x["score"], reverse=True)

    if alarm_filter == "soft" and alarm_id is not None:
        wanted_alarm = str(alarm_id).strip()
        # Keep semantic ordering but bias toward same-alarm candidates.
        for c in candidates:
            c_alarm = str(c.get("meta", {}).get("alarm_id", "")).strip()
            c["score_with_alarm_bias"] = c["score"] + (0.05 if c_alarm == wanted_alarm else 0.0)
        candidates.sort(key=lambda x: x["score_with_alarm_bias"], reverse=True)

    top_children = candidates[:top_k]

    # Parent-child context restoration
    parent_ids = []
    for c in top_children:
        pid = c["meta"]["parent_id"]
        if pid not in parent_ids:
            parent_ids.append(pid)
    parents = [kb["parents"][pid] for pid in parent_ids]

    return {
        "query": query,
        "component_id": component_id,
        "children": top_children,
        "parents": parents,
    }


def cmd_build(args: argparse.Namespace) -> None:
    build_kb(
        qa_jsonl=Path(args.qa_jsonl),
        out_dir=Path(args.out_dir),
        bi_encoder_name=args.bi_encoder,
        embed_backend=args.embed_backend,
        ollama_embed_model=args.ollama_embed_model,
        ollama_url=args.ollama_url,
        parent_tokens=args.parent_tokens,
        child_tokens=args.child_tokens,
        child_overlap=args.child_overlap,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        batch_size=args.batch_size,
    )


def cmd_query(args: argparse.Namespace) -> None:
    kb_dir = Path(args.kb_dir)
    kb = load_kb(kb_dir)
    embed_backend = args.embed_backend or kb["config"].get("embed_backend", "hf")
    bi_encoder_name = args.bi_encoder or kb["config"].get("bi_encoder")
    if embed_backend == "hf":
        _lazy_import_sentence_transformers()
        bi_encoder = SentenceTransformer(bi_encoder_name)
    else:
        ollama_model = args.ollama_embed_model or kb["config"].get("ollama_embed_model") or "nomic-embed-text"
        ollama_url = args.ollama_url or kb["config"].get("ollama_url") or "http://127.0.0.1:11434"
        bi_encoder = OllamaEmbedder(model=ollama_model, ollama_url=ollama_url)

    if args.use_rerank:
        _lazy_import_sentence_transformers()
        reranker = CrossEncoder(args.cross_encoder)
    else:
        reranker = None

    result = retrieve(
        kb=kb,
        bi_encoder=bi_encoder,
        query=args.query,
        component_id=args.component_id,
        alarm_id=None,
        alarm_filter="off",
        top_k=args.top_k,
        candidate_k=max(args.top_k, args.candidate_k),
        use_rerank=args.use_rerank,
        reranker=reranker,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline TKBA builder/query tool for wind-turbine QA data.")
    sub = p.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build an offline TKBA index from QA JSONL.")
    p_build.add_argument("--qa-jsonl", required=True, help="Input QA dataset JSONL.")
    p_build.add_argument("--out-dir", required=True, help="Output directory for index and metadata artifacts.")
    p_build.add_argument("--embed-backend", choices=["hf", "ollama"], default="hf")
    p_build.add_argument("--bi-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p_build.add_argument("--ollama-embed-model", default="nomic-embed-text")
    p_build.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--parent-tokens", type=int, default=1024)
    p_build.add_argument("--child-tokens", type=int, default=128)
    p_build.add_argument("--child-overlap", type=int, default=24)
    p_build.add_argument("--hnsw-m", type=int, default=32)
    p_build.add_argument("--ef-construction", type=int, default=200)
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Query an existing offline TKBA index.")
    p_query.add_argument("--kb-dir", required=True, help="Directory produced by the build command.")
    p_query.add_argument("--query", required=True, help="Natural-language retrieval query.")
    p_query.add_argument("--component-id", default=None, help="Hard metadata filter: exact component_id")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--candidate-k", type=int, default=50)
    p_query.add_argument("--embed-backend", choices=["hf", "ollama"], default=None)
    p_query.add_argument("--bi-encoder", default=None, help="Override bi-encoder model")
    p_query.add_argument("--ollama-embed-model", default=None)
    p_query.add_argument("--ollama-url", default=None)
    p_query.add_argument("--use-rerank", action="store_true")
    p_query.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p_query.set_defaults(func=cmd_query)
    return p


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
