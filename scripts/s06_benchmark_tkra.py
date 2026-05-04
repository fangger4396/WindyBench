#!/usr/bin/env python3
"""
Run benchmark for multiple methods and output ROUGE comparison table.

Methods supported:
- base: question-only generation
- sft: question-only generation with stronger instruction
- rag: question + retrieved context
- rag_hyde: rag + HyDE query expansion
- rag_hyde_rerank: rag + HyDE + rerank

Outputs:
- predictions per method: <out-dir>/<method>.pred.jsonl
- summary csv/json: <out-dir>/benchmark_results.csv/.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from rouge_score import rouge_scorer

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
AutoTokenizer = None
AutoModelForCausalLM = None
torch = None
_HF_MODEL_CACHE: Dict[str, Any] = {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def call_ollama(model: str, prompt: str, timeout_sec: int = 180, temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9},
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


def _lazy_import_transformers_torch() -> None:
    global AutoTokenizer, AutoModelForCausalLM, torch
    if AutoTokenizer is not None and AutoModelForCausalLM is not None and torch is not None:
        return
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM, AutoTokenizer as _AutoTokenizer

    torch = _torch
    AutoTokenizer = _AutoTokenizer
    AutoModelForCausalLM = _AutoModelForCausalLM


def parse_model_spec(spec: str) -> Dict[str, str]:
    # Supported:
    # - ollama:llama2:7b
    # - hf:/data_hdd/.../llama-7b-merged-cot
    # Backward-compatible fallback: plain string -> ollama model name
    if spec.startswith("ollama:"):
        return {"backend": "ollama", "model": spec[len("ollama:") :]}
    if spec.startswith("hf:"):
        return {"backend": "hf", "model": spec[len("hf:") :]}
    return {"backend": "ollama", "model": spec}


def _get_hf_model(model_path: str, verbose: bool = False):
    if model_path in _HF_MODEL_CACHE:
        if verbose:
            print(f"[hf] reuse cached model: {model_path}")
        return _HF_MODEL_CACHE[model_path]
    _lazy_import_transformers_torch()
    local_only = Path(model_path).exists()
    if verbose:
        print(f"[hf] loading model from: {model_path} (local_files_only={local_only})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_only,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    if verbose:
        try:
            dmap = getattr(model, "hf_device_map", None)
            print(f"[hf] device_map: {dmap}")
        except Exception:
            pass
    _HF_MODEL_CACHE[model_path] = (tokenizer, model)
    return tokenizer, model


def call_hf_local(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 320,
    temperature: float = 0.1,
    verbose: bool = False,
) -> str:
    tokenizer, model = _get_hf_model(model_path, verbose=verbose)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attn = inputs["attention_mask"].to(model.device)
    do_sample = temperature > 1e-6
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-6),
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][input_ids.shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def generate_with_spec(
    model_spec: str,
    prompt: str,
    timeout_sec: int,
    temperature: float = 0.1,
    max_new_tokens: int = 320,
    verbose: bool = False,
) -> str:
    info = parse_model_spec(model_spec)
    if info["backend"] == "ollama":
        if verbose:
            print(f"[gen] backend=ollama model={info['model']}")
        return call_ollama(info["model"], prompt, timeout_sec=timeout_sec, temperature=temperature)
    if info["backend"] == "hf":
        if verbose:
            print(f"[gen] backend=hf model={info['model']}")
        return call_hf_local(
            info["model"], prompt, max_new_tokens=max_new_tokens, temperature=temperature, verbose=verbose
        )
    raise ValueError(f"Unknown model backend in spec: {model_spec}")


def normalize_text(s: str) -> str:
    return " ".join(str(s).strip().split())


def tokenize_text(s: str) -> List[str]:
    return normalize_text(s).lower().split()


def eval_text_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Generation quality metrics without extra labels.

    - accuracy: exact normalized string match
    - precision/recall/f1: token-overlap micro averages
    - macro_f1: average sample-level F1
    """
    exact = 0
    tp = 0
    pred_total = 0
    ref_total = 0
    f1s: List[float] = []
    for r in rows:
        pred = tokenize_text(r["prediction"])
        ref = tokenize_text(r["reference"])
        if normalize_text(r["prediction"]) == normalize_text(r["reference"]):
            exact += 1
        pred_total += len(pred)
        ref_total += len(ref)
        pred_set = set(pred)
        ref_set = set(ref)
        inter = len(pred_set & ref_set)
        tp += inter
        p = inter / max(1, len(pred_set))
        rec = inter / max(1, len(ref_set))
        f1 = (2 * p * rec / (p + rec)) if (p + rec) > 0 else 0.0
        f1s.append(f1)
    n = max(1, len(rows))
    micro_p = tp / max(1, pred_total)
    micro_r = tp / max(1, ref_total)
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    return {
        "accuracy": exact / n,
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f1,
        "macro_f1": sum(f1s) / n,
    }


def _ndcg_at_k_binary(labels: List[int]) -> float:
    # labels are 0/1 relevance at ranking positions 1..k
    if not labels:
        return 0.0
    dcg = 0.0
    for i, rel in enumerate(labels, start=1):
        if rel > 0:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sorted(labels, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal, start=1):
        if rel > 0:
            idcg += 1.0 / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def eval_retrieval_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    # TKRA-oriented retrieval diagnostics:
    # these metrics quantify how well retrieval aligns with expected alarm/component evidence.
    rag_rows = [r for r in rows if "retrieval_k" in r]
    if not rag_rows:
        return {}

    def avg(key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in rag_rows]
        return sum(vals) / max(1, len(vals))

    return {
        "retrieval_count": len(rag_rows),
        "component_consistency_rate": avg("retrieval_top1_component_match"),
        "alarm_top1_accuracy": avg("retrieval_top1_alarm_match"),
        "recall_at_k_alarm": avg("retrieval_recall_at_k_alarm"),
        "precision_at_k_alarm": avg("retrieval_precision_at_k_alarm"),
        "mrr_alarm": avg("retrieval_mrr_alarm"),
        "ndcg_at_k_alarm": avg("retrieval_ndcg_at_k_alarm"),
        "recall_at_k_component": avg("retrieval_recall_at_k_component"),
        "precision_at_k_component": avg("retrieval_precision_at_k_component"),
        "mrr_component": avg("retrieval_mrr_component"),
        "ndcg_at_k_component": avg("retrieval_ndcg_at_k_component"),
    }


def eval_rouge(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for r in rows:
        pred = normalize_text(r["prediction"])
        ref = normalize_text(r["reference"])
        s = scorer.score(ref, pred)
        r1.append(float(s["rouge1"].fmeasure))
        r2.append(float(s["rouge2"].fmeasure))
        rl.append(float(s["rougeL"].fmeasure))
    n = max(1, len(rows))
    return {
        "count": len(rows),
        "rouge1": sum(r1) / n,
        "rouge2": sum(r2) / n,
        "rougeL": sum(rl) / n,
    }


def make_base_prompt(question: str) -> str:
    return (
        "You are a wind turbine diagnosis assistant.\n"
        "Answer using exactly three sections:\n"
        "Cause Analysis:\nAction Recommendations:\nRisk Evaluation:\n\n"
        f"Question:\n{question}\n"
    )


def make_sft_prompt(question: str) -> str:
    return (
        "You are an expert wind turbine O&M engineer. "
        "Use technical, concise, evidence-oriented language.\n"
        "Output exactly three sections:\n"
        "Cause Analysis:\nAction Recommendations:\nRisk Evaluation:\n\n"
        f"Question:\n{question}\n"
    )


def build_evidence_block(row: Dict[str, Any], full_evidence_in_prompt: bool) -> str:
    if not full_evidence_in_prompt:
        return ""
    manifest = row.get("manifest", {})
    evidence = row.get("evidence", [])
    return (
        "\nManifest:\n"
        + json.dumps(manifest, ensure_ascii=False)
        + "\n\nFull SCADA Evidence:\n"
        + json.dumps(evidence, ensure_ascii=False)
        + "\n"
    )


def make_hyde_query(llm_model: str, manifest: Dict[str, Any]) -> str:
    p = (
        "Write a short technical diagnostic paragraph for retrieval expansion.\n"
        f"Manifest:\n{json.dumps(manifest, ensure_ascii=False)}"
    )
    return generate_with_spec(llm_model, p, timeout_sec=180, temperature=0.0, max_new_tokens=220)


def make_rag_prompt(
    question: str,
    manifest: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    evidence_block: str = "",
) -> str:
    ctext = "\n\n".join([f"Evidence {i+1}:\n{clean_retrieved_text(c.get('text', ''))}" for i, c in enumerate(contexts)])
    return (
        "You are an expert wind turbine O&M engineer.\n"
        "Use retrieved evidence as grounding, but do not copy metadata blocks or source tags.\n"
        "Strict output rules:\n"
        "1) Output exactly three sections in this order.\n"
        "2) Section headers must be exactly:\n"
        "Cause Analysis:\n"
        "Action Recommendations:\n"
        "Risk Evaluation:\n"
        "3) Do not output source tags such as [C1]/Evidence 1/SampleID/AlarmID/EventStart/EventEnd.\n"
        "4) Synthesize concise technical diagnosis, not raw retrieval snippets.\n\n"
        f"Question:\n{question}\n\n"
        + evidence_block
        + "\n"
        f"Manifest:\n{json.dumps(manifest, ensure_ascii=False)}\n\n"
        f"Retrieved Evidence:\n{ctext}\n\n"
        "Final Answer:\n"
        "Cause Analysis:\n"
    )


def clean_retrieved_text(text: str) -> str:
    """Remove KB metadata-like lines to reduce copy-over in generation."""
    s = str(text or "")
    drop_prefixes = (
        "sampleid:",
        "turbineid:",
        "alarmid:",
        "alarmdesc:",
        "componentid:",
        "eventstart:",
        "eventend:",
        "symptommanifest:",
    )
    kept: List[str] = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t.lower().startswith(drop_prefixes):
            continue
        if re.match(r"^\[c\d+\]\s*", t.lower()):
            t = re.sub(r"^\[c\d+\]\s*", "", t, flags=re.IGNORECASE)
            if not t:
                continue
        kept.append(t)
    out = "\n".join(kept)
    return out if out else s


def run_method(
    method: str,
    rows: List[Dict[str, Any]],
    model_spec: str,
    kb: Optional[Dict[str, Any]],
    bi_encoder: Optional[Any],
    reranker: Optional[Any],
    top_k: int,
    candidate_k: int,
    timeout_sec: int,
    max_new_tokens: int,
    progress_every: int,
    verbose: bool,
    full_evidence_in_prompt: bool,
    alarm_filter: str,
    rag_fallback_base: bool,
    rag_min_score: float,
    rag_require_alarm_match: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    use_rag = method in {"rag", "rag_hyde", "rag_hyde_rerank"}
    use_hyde = method in {"rag_hyde", "rag_hyde_rerank"}
    use_rerank = method == "rag_hyde_rerank"

    latencies: List[float] = []
    for i, row in enumerate(rows, start=1):
        q = str(row.get("question", ""))
        ref = str(row.get("answer", ""))
        manifest = row.get("manifest", {})
        sample_id = row.get("sample_id", f"row_{i}")
        evidence_block = build_evidence_block(row, full_evidence_in_prompt=full_evidence_in_prompt)

        if not use_rag:
            prompt = (make_base_prompt(q) if method == "base" else make_sft_prompt(q)) + evidence_block
            t0 = time.time()
            pred = generate_with_spec(
                model_spec,
                prompt,
                timeout_sec=timeout_sec,
                temperature=0.1,
                max_new_tokens=max_new_tokens,
                verbose=verbose and i == 1,
            )
            latencies.append(time.time() - t0)
        else:
            assert kb is not None and bi_encoder is not None
            alarm_id = str(row.get("alarm_id", ""))
            alarm_desc = str(row.get("alarm_desc", ""))
            query = (
                make_hyde_query(model_spec, manifest)
                if use_hyde
                else (
                    f"{manifest.get('component_id','')} "
                    f"AlarmID:{alarm_id} "
                    f"{alarm_desc} "
                    f"{manifest.get('trend_description','')} "
                    f"{q}"
                )
            )
            ret = retrieve(
                kb=kb,
                bi_encoder=bi_encoder,
                query=query,
                component_id=manifest.get("component_id"),
                alarm_id=alarm_id,
                alarm_filter=alarm_filter,
                top_k=top_k,
                candidate_k=max(top_k, candidate_k),
                use_rerank=use_rerank,
                reranker=reranker if use_rerank else None,
            )
            # Prefer top retrieved child chunks to reduce irrelevant parent-level noise.
            contexts = ret.get("children", [])[:top_k]
            if not contexts:
                contexts = ret.get("parents", [])[:top_k]
            use_rag_context = True
            top1 = contexts[0] if contexts else None
            top1_score = float(top1.get("score", -1e9)) if top1 else -1e9
            top1_alarm = str(top1.get("meta", {}).get("alarm_id", "")) if top1 else ""
            target_component = str(manifest.get("component_id", ""))
            labels_alarm: List[int] = []
            labels_component: List[int] = []
            for c in contexts:
                cm = c.get("meta", {})
                ca = str(cm.get("alarm_id", ""))
                cc = str(cm.get("component_id", ""))
                labels_alarm.append(1 if (alarm_id and ca == alarm_id) else 0)
                labels_component.append(1 if (target_component and cc == target_component) else 0)

            if not contexts:
                use_rag_context = False
            if rag_min_score > -1e8 and top1_score < rag_min_score:
                use_rag_context = False
            if rag_require_alarm_match and alarm_id and top1_alarm and top1_alarm != alarm_id:
                use_rag_context = False

            if rag_fallback_base and not use_rag_context:
                prompt = make_sft_prompt(q) + evidence_block
            else:
                prompt = make_rag_prompt(q, manifest, contexts, evidence_block=evidence_block)
            t0 = time.time()
            pred = generate_with_spec(
                model_spec,
                prompt,
                timeout_sec=timeout_sec,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                verbose=verbose and i == 1,
            )
            latencies.append(time.time() - t0)

        row_out = {"sample_id": sample_id, "method": method, "prediction": pred, "reference": ref}
        if use_rag:
            k_eff = len(contexts)
            first_alarm_rank = next((idx + 1 for idx, v in enumerate(labels_alarm) if v == 1), None)
            first_component_rank = next((idx + 1 for idx, v in enumerate(labels_component) if v == 1), None)
            row_out.update(
                {
                    "retrieval_k": k_eff,
                    "retrieval_top1_score": top1_score if contexts else 0.0,
                    "retrieval_top1_alarm_match": 1.0 if (contexts and labels_alarm[0] == 1) else 0.0,
                    "retrieval_top1_component_match": 1.0 if (contexts and labels_component[0] == 1) else 0.0,
                    "retrieval_recall_at_k_alarm": 1.0 if any(labels_alarm) else 0.0,
                    "retrieval_precision_at_k_alarm": (sum(labels_alarm) / max(1, k_eff)) if k_eff > 0 else 0.0,
                    "retrieval_mrr_alarm": (1.0 / first_alarm_rank) if first_alarm_rank else 0.0,
                    "retrieval_ndcg_at_k_alarm": _ndcg_at_k_binary(labels_alarm),
                    "retrieval_recall_at_k_component": 1.0 if any(labels_component) else 0.0,
                    "retrieval_precision_at_k_component": (sum(labels_component) / max(1, k_eff))
                    if k_eff > 0
                    else 0.0,
                    "retrieval_mrr_component": (1.0 / first_component_rank) if first_component_rank else 0.0,
                    "retrieval_ndcg_at_k_component": _ndcg_at_k_binary(labels_component),
                }
            )
        out.append(row_out)
        if i % max(1, progress_every) == 0:
            avg_lat = sum(latencies) / max(1, len(latencies))
            print(f"[{method}] done={i}/{len(rows)} avg_sec={avg_lat:.2f}")
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run benchmark and compute ROUGE comparison.")
    ap.add_argument("--test-jsonl", required=True, help="Test rows containing question/answer/manifest")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--methods", default="base,sft,rag,rag_hyde,rag_hyde_rerank")
    ap.add_argument(
        "--base-model",
        default="ollama:llama2:7b",
        help="Model spec: ollama:<name> or hf:</local/path>. Plain name = ollama name.",
    )
    ap.add_argument(
        "--sft-model",
        default="ollama:llama2:7b",
        help="Model spec: ollama:<name> or hf:</local/path>. Plain name = ollama name.",
    )
    ap.add_argument(
        "--rag-model",
        default="ollama:llama2:7b",
        help="Model spec: ollama:<name> or hf:</local/path>. Plain name = ollama name.",
    )
    ap.add_argument("--kb-dir", default=None, help="Required for rag methods")
    ap.add_argument("--embed-backend", choices=["auto", "hf", "ollama"], default="auto")
    ap.add_argument("--bi-encoder", default=None, help="Override KB bi-encoder")
    ap.add_argument("--ollama-embed-model", default=None)
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument(
        "--alarm-filter",
        choices=["off", "soft", "hard"],
        default="off",
        help="Alarm-level retrieval constraint: off=none, soft=score bias, hard=strict filter when available.",
    )
    ap.add_argument(
        "--rag-fallback-base",
        action="store_true",
        help="If retrieval is low-confidence, fallback to no-context prompt for rag methods.",
    )
    ap.add_argument(
        "--rag-min-score",
        type=float,
        default=-1.0,
        help="Top1 retrieval score threshold for using RAG context. Set < -1 to disable.",
    )
    ap.add_argument(
        "--rag-require-alarm-match",
        action="store_true",
        help="Require top1 retrieved alarm_id to match current row alarm_id, else fallback.",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--progress-every", type=int, default=20, help="Print progress every N samples")
    ap.add_argument("--verbose", action="store_true", help="Verbose model/backend debug logs")
    ap.add_argument(
        "--full-evidence-in-prompt",
        action="store_true",
        help="Inject full manifest+evidence (including series_points) from test row into each prompt.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.test_jsonl))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    need_rag = any(m in {"rag", "rag_hyde", "rag_hyde_rerank"} for m in methods)
    kb = None
    bi_encoder = None
    reranker = None
    if need_rag:
        if not args.kb_dir:
            raise ValueError("RAG methods require --kb-dir")
        kb = load_kb(Path(args.kb_dir))
        backend = args.embed_backend
        if backend == "auto":
            backend = kb["config"].get("embed_backend", "hf")
        if backend == "hf":
            _lazy_import_sentence_transformers()
            bi_encoder_name = args.bi_encoder or kb["config"]["bi_encoder"]
            bi_encoder = SentenceTransformer(bi_encoder_name)
        else:
            ollama_model = args.ollama_embed_model or kb["config"].get("ollama_embed_model") or "nomic-embed-text"
            ollama_url = args.ollama_url or kb["config"].get("ollama_url") or "http://127.0.0.1:11434"
            bi_encoder = OllamaEmbedder(model=ollama_model, ollama_url=ollama_url)
        if "rag_hyde_rerank" in methods:
            _lazy_import_sentence_transformers()
            reranker = CrossEncoder(args.cross_encoder)

    table_rows: List[Dict[str, Any]] = []
    for m in methods:
        model_spec = args.base_model
        if m == "sft":
            model_spec = args.sft_model
        if m in {"rag", "rag_hyde", "rag_hyde_rerank"}:
            model_spec = args.rag_model

        pred_rows = run_method(
            method=m,
            rows=rows,
            model_spec=model_spec,
            kb=kb,
            bi_encoder=bi_encoder,
            reranker=reranker,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            timeout_sec=args.timeout_sec,
            max_new_tokens=args.max_new_tokens,
            progress_every=args.progress_every,
            verbose=args.verbose,
            full_evidence_in_prompt=args.full_evidence_in_prompt,
            alarm_filter=args.alarm_filter,
            rag_fallback_base=args.rag_fallback_base,
            rag_min_score=args.rag_min_score,
            rag_require_alarm_match=args.rag_require_alarm_match,
        )
        pred_path = out_dir / f"{m}.pred.jsonl"
        write_jsonl(pred_path, pred_rows)
        metrics = eval_rouge(pred_rows)
        metrics.update(eval_text_metrics(pred_rows))
        metrics.update(eval_retrieval_metrics(pred_rows))
        metrics["method"] = m
        metrics["pred_file"] = str(pred_path)
        table_rows.append(metrics)
        msg = (
            f"{m}: R1={metrics['rouge1']:.4f} R2={metrics['rouge2']:.4f} RL={metrics['rougeL']:.4f} "
            f"F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f}"
        )
        if "mrr_alarm" in metrics:
            msg += f" MRR(alarm)={metrics['mrr_alarm']:.4f}"
        print(msg)

    (out_dir / "benchmark_results.json").write_text(
        json.dumps(table_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "benchmark_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "count",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "macro_f1",
                "rouge1",
                "rouge2",
                "rougeL",
                "retrieval_count",
                "component_consistency_rate",
                "alarm_top1_accuracy",
                "recall_at_k_alarm",
                "precision_at_k_alarm",
                "mrr_alarm",
                "ndcg_at_k_alarm",
                "recall_at_k_component",
                "precision_at_k_component",
                "mrr_component",
                "ndcg_at_k_component",
                "pred_file",
            ]
        )
        for r in table_rows:
            w.writerow(
                [
                    r.get("method"),
                    r.get("count"),
                    r.get("accuracy", ""),
                    r.get("precision", ""),
                    r.get("recall", ""),
                    r.get("f1", ""),
                    r.get("macro_f1", ""),
                    r.get("rouge1"),
                    r.get("rouge2"),
                    r.get("rougeL"),
                    r.get("retrieval_count", ""),
                    r.get("component_consistency_rate", ""),
                    r.get("alarm_top1_accuracy", ""),
                    r.get("recall_at_k_alarm", ""),
                    r.get("precision_at_k_alarm", ""),
                    r.get("mrr_alarm", ""),
                    r.get("ndcg_at_k_alarm", ""),
                    r.get("recall_at_k_component", ""),
                    r.get("precision_at_k_component", ""),
                    r.get("mrr_component", ""),
                    r.get("ndcg_at_k_component", ""),
                    r.get("pred_file"),
                ]
            )
    print(f"Saved: {out_dir/'benchmark_results.csv'}")


if __name__ == "__main__":
    main()
