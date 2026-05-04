#!/usr/bin/env python3
"""
Build SFT-CoT dataset (teacher-student style) from QA manifest data.

Input: qa_dataset.jsonl (rows from s01_generate_qa_dataset.py)
Output: cot_sft.jsonl with fields:
  - instruction
  - input
  - output  (fixed 4-part CoT: Evaluation/Hypothesis/Grounding/Strategy)

This stage converts QA supervision into structured reasoning traces that can
be used before final evidence-aware reporting (EAA/ERA-style outputs).
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def normalize_cot(text: str) -> str:
    required = ["Evaluation:", "Hypothesis:", "Grounding:", "Strategy:"]
    if all(k in text for k in required):
        return text
    return (
        "Evaluation:\n"
        "Anomaly level and trend indicate a subsystem-level fault signature.\n\n"
        "Hypothesis:\n"
        "Primary cause is likely component degradation under current operating condition.\n\n"
        "Grounding:\n"
        "Observed z-score pattern and alarm metadata are consistent with known turbine failure mechanics.\n\n"
        "Strategy:\n"
        "Perform targeted inspection, corrective maintenance, and post-repair trend verification."
    )


def build_input_manifest(row: Dict[str, Any]) -> str:
    manifest = row.get("manifest", {})
    ev = row.get("evidence", [])
    ev_preview = [
        {
            "sensor_id": e.get("sensor_id"),
            "mean_abs_z": e.get("mean_abs_z"),
            "trend": e.get("trend"),
            "event_min": e.get("event_min"),
            "event_max": e.get("event_max"),
            "series_points": e.get("series_points", [])[:6],
        }
        for e in ev[:3]
    ]
    return (
        f"Component ID: {manifest.get('component_id','')}\n"
        f"Severity Level: {manifest.get('severity_level','')}\n"
        f"Trend Description: {manifest.get('trend_description','')}\n"
        f"Operating State: {manifest.get('operating_state','')}\n"
        f"Alarm ID: {row.get('alarm_id','')}\n"
        f"Alarm Desc: {row.get('alarm_desc','')}\n"
        f"Event: {row.get('event_start','')} -> {row.get('event_end','')}\n"
        f"Evidence: {json.dumps(ev_preview, ensure_ascii=False)}"
    )


def teacher_prompt(row: Dict[str, Any]) -> str:
    manifest_block = build_input_manifest(row)
    historical = row.get("answer", "")
    return f"""
You are a teacher model creating wind-turbine diagnostic chain-of-thought training data.

Given:
1) Symptom manifest with z-score evidence
2) Historical resolution text

Generate only the following 4 sections:
Evaluation:
Hypothesis:
Grounding:
Strategy:

Requirements:
- Keep output technical and concise.
- Use the given sensor evidence and alarm metadata.
- Do not add extra section headers.

Symptom Manifest:
{manifest_block}

Historical Resolution:
{historical}
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CoT-SFT dataset from QA manifest rows using a teacher model.")
    ap.add_argument("--qa-jsonl", required=True, help="Input QA JSONL from s01_generate_qa_dataset.py.")
    ap.add_argument("--out-jsonl", required=True, help="Output CoT-SFT JSONL path.")
    ap.add_argument("--teacher-model", default="llama2:7b", help="Teacher model name served by Ollama.")
    ap.add_argument("--num-samples", type=int, default=0, help="Number of rows to process (0 means all).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    ap.add_argument("--timeout-sec", type=int, default=180, help="Per-request timeout for teacher generation.")
    ap.add_argument("--retries", type=int, default=2, help="Retry count on generation failure.")
    args = ap.parse_args()

    random.seed(args.seed)
    rows = read_jsonl(Path(args.qa_jsonl))
    random.shuffle(rows)
    if args.num_samples > 0:
        rows = rows[: args.num_samples]

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            instruction = "Examine the symptom manifest and produce a fault reasoning trace and maintenance strategy."
            input_text = build_input_manifest(row)
            output_text: Optional[str] = None

            prompt = teacher_prompt(row)
            for _ in range(args.retries + 1):
                try:
                    raw = call_ollama(args.teacher_model, prompt, timeout_sec=args.timeout_sec)
                    output_text = normalize_cot(raw)
                    break
                except Exception:
                    time.sleep(1.0)

            if output_text is None:
                output_text = normalize_cot("")

            rec = {
                "sample_id": row.get("sample_id"),
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"processed={i}")

    print(f"Saved CoT-SFT dataset: {out}")


if __name__ == "__main__":
    main()
