#!/usr/bin/env python3
"""
BDMA-style multi-agent inference pipeline:
SPA -> CRA(TKBA index) -> DSA(EAA generation)

Input: manifest JSON (or generated from qa row)
Output: structured report:
  - Cause Analysis
  - Action Recommendations
  - Risk Evaluation
  - Verification Citations
"""

# Acronym mapping used in this repository:
# - SPA: Symptom Parsing Agent (manifest construction / symptom normalization)
# - CRA: Context Retrieval Agent (offline TKBA retrieval with optional rerank/HyDE)
# - TKBA: Turbine Knowledge Base Archive (offline parent-child chunk index)
# - TKRA: Turbine Knowledge Retrieval Augmentation (runtime retrieval augmentation behavior)
# - DSA: Diagnostic Synthesis Agent (final report generation)
# - EAA/ERA: Evidence-Aware Analysis / Evidence-based Risk Assessment style outputs

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from s05_build_tkba_index import OllamaEmbedder, load_kb, retrieve

SentenceTransformer = None
CrossEncoder = None


def _lazy_import_sentence_transformers() -> None:
    global SentenceTransformer, CrossEncoder
    if SentenceTransformer is not None and CrossEncoder is not None:
        return
    from sentence_transformers import CrossEncoder as _CrossEncoder, SentenceTransformer as _SentenceTransformer

    SentenceTransformer = _SentenceTransformer
    CrossEncoder = _CrossEncoder


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def call_ollama(model: str, prompt: str, timeout_sec: int = 180) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return str(obj.get("response", "")).strip()


def _extract_section(text: str, header: str, next_headers: List[str]) -> str:
    low = text
    start = low.find(header)
    if start < 0:
        return ""
    start += len(header)
    end = len(text)
    for h in next_headers:
        idx = low.find(h, start)
        if idx >= 0:
            end = min(end, idx)
    return text[start:end].strip()


def normalize_report(text: str, citations: List[Dict[str, Any]], strict_output: bool = False) -> str:
    need = ["Cause Analysis:", "Action Recommendations:", "Risk Evaluation:"]
    if not all(k in text for k in need):
        text = (
            "Cause Analysis:\n"
            "Evidence suggests subsystem-level fault progression under current operating condition.\n\n"
            "Action Recommendations:\n"
            "Execute targeted inspection, corrective maintenance, and trend-back verification.\n\n"
            "Risk Evaluation:\n"
            "If unresolved, risk of unplanned downtime and secondary component damage increases."
        )
    elif strict_output:
        cause = _extract_section(text, "Cause Analysis:", ["Action Recommendations:", "Risk Evaluation:"])
        action = _extract_section(text, "Action Recommendations:", ["Risk Evaluation:"])
        risk = _extract_section(text, "Risk Evaluation:", [])
        text = (
            "Cause Analysis:\n"
            + (cause or "Evidence indicates subsystem-level fault progression.")
            + "\n\nAction Recommendations:\n"
            + (action or "Perform targeted inspection, corrective maintenance, and post-repair verification.")
            + "\n\nRisk Evaluation:\n"
            + (risk or "If unresolved, fault escalation may increase downtime and secondary damage risk.")
        )
    cite_lines = []
    for i, c in enumerate(citations, start=1):
        meta = c.get("metadata", {})
        cite_lines.append(
            f"[{i}] sample_id={meta.get('sample_id','')}, alarm_id={meta.get('alarm_id','')}, "
            f"component={c.get('component_id','')}"
        )
    return text + "\n\nVerification Citations:\n" + ("\n".join(cite_lines) if cite_lines else "None")


def make_hyde_query(llm_model: str, manifest: Dict[str, Any]) -> str:
    prompt = f"""
Write a concise technical diagnostic paragraph for retrieval expansion.
Focus on mechanical/electrical failure semantics, maintenance terms, and component behavior.

Manifest:
{json.dumps(manifest, ensure_ascii=False, indent=2)}
""".strip()
    return call_ollama(llm_model, prompt)


def build_dsa_prompt(manifest: Dict[str, Any], contexts: List[Dict[str, Any]]) -> str:
    context_text = []
    for i, c in enumerate(contexts, start=1):
        context_text.append(f"[Context {i}] {c.get('text','')}")
    return f"""
You are the Diagnostic Synthesis Agent (DSA).
Use the manifest and retrieved technical context to output:
Cause Analysis:
Action Recommendations:
Risk Evaluation:

Rules:
- Ground reasoning in retrieved context.
- Keep concise and actionable.
- Do not invent component names not in manifest/context.

Manifest:
{json.dumps(manifest, ensure_ascii=False, indent=2)}

Retrieved Context:
{chr(10).join(context_text)}
""".strip()


def read_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    if args.manifest_json:
        return json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    if args.qa_row_json:
        row = json.loads(Path(args.qa_row_json).read_text(encoding="utf-8"))
        return {
            "component_id": row.get("manifest", {}).get("component_id"),
            "severity_level": row.get("manifest", {}).get("severity_level"),
            "mean_abs_z_primary": row.get("manifest", {}).get("mean_abs_z_primary"),
            "trend_description": row.get("manifest", {}).get("trend_description"),
            "operating_state": row.get("manifest", {}).get("operating_state"),
            "alarm_id": row.get("alarm_id"),
            "alarm_desc": row.get("alarm_desc"),
            "event_start": row.get("event_start"),
            "event_end": row.get("event_end"),
        }
    if args.manifest_inline:
        return json.loads(args.manifest_inline)
    raise ValueError("Provide one of --manifest-json / --qa-row-json / --manifest-inline")


def main() -> None:
    ap = argparse.ArgumentParser(description="BDMA-style multi-agent inference (SPA -> CRA -> DSA).")
    ap.add_argument("--kb-dir", required=True, help="Offline TKBA directory generated by s05_build_tkba_index.py.")
    ap.add_argument("--manifest-json", default=None, help="Path to a manifest JSON file.")
    ap.add_argument("--qa-row-json", default=None, help="Path to a JSON file containing one QA row.")
    ap.add_argument("--manifest-inline", default=None, help="Inline manifest JSON string.")
    ap.add_argument("--llm-model", default="llama2:7b", help="LLM model name served by Ollama.")
    ap.add_argument("--embed-backend", choices=["hf", "ollama"], default=None, help="Retriever embedding backend.")
    ap.add_argument("--bi-encoder", default=None, help="Override KB bi-encoder model name.")
    ap.add_argument("--ollama-embed-model", default=None, help="Override Ollama embedding model name.")
    ap.add_argument("--ollama-url", default=None, help="Override Ollama base URL.")
    ap.add_argument("--use-rerank", action="store_true", help="Enable cross-encoder reranking in CRA.")
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Reranker model name.")
    ap.add_argument("--top-k", type=int, default=5, help="Number of retrieved contexts used by DSA.")
    ap.add_argument("--candidate-k", type=int, default=50, help="Candidate pool size before rerank/top-k cut.")
    ap.add_argument("--hyde", action="store_true", help="Enable HyDE query expansion for CRA retrieval.")
    ap.add_argument("--strict-output", action="store_true", help="Keep only the 3 required sections in final report")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    manifest = read_manifest(args)
    kb = load_kb(Path(args.kb_dir))
    embed_backend = args.embed_backend or kb["config"].get("embed_backend", "hf")
    if embed_backend == "hf":
        _lazy_import_sentence_transformers()
        bi_name = args.bi_encoder or kb["config"]["bi_encoder"]
        bi_encoder = SentenceTransformer(bi_name)
    else:
        ollama_model = args.ollama_embed_model or kb["config"].get("ollama_embed_model") or "nomic-embed-text"
        ollama_url = args.ollama_url or kb["config"].get("ollama_url") or "http://127.0.0.1:11434"
        bi_encoder = OllamaEmbedder(model=ollama_model, ollama_url=ollama_url)

    if args.use_rerank:
        _lazy_import_sentence_transformers()
        reranker = CrossEncoder(args.cross_encoder)
    else:
        reranker = None

    # CRA query
    base_query = (
        f"{manifest.get('component_id','')} {manifest.get('alarm_desc','')} "
        f"{manifest.get('trend_description','')} z={manifest.get('mean_abs_z_primary','')}"
    )
    query = make_hyde_query(args.llm_model, manifest) if args.hyde else base_query

    retrieval = retrieve(
        kb=kb,
        bi_encoder=bi_encoder,
        query=query,
        component_id=manifest.get("component_id"),
        top_k=args.top_k,
        candidate_k=max(args.top_k, args.candidate_k),
        use_rerank=args.use_rerank,
        reranker=reranker,
    )

    contexts = retrieval.get("parents", [])[: args.top_k]
    dsa_prompt = build_dsa_prompt(manifest, contexts)
    raw_report = call_ollama(args.llm_model, dsa_prompt)
    final_report = normalize_report(raw_report, contexts, strict_output=args.strict_output)

    out = {
        "manifest": manifest,
        "query_used": query,
        "retrieval_children": retrieval.get("children", []),
        "retrieval_parents": contexts,
        "final_report": final_report,
    }

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {args.out_json}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
