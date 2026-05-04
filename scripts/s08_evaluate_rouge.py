#!/usr/bin/env python3
"""
Compute ROUGE-1/ROUGE-2/ROUGE-L between predictions and references.

This is the lightweight post-hoc evaluator for generated diagnostic reports.
It expects one JSON object per line and outputs aggregate metrics.

Input JSONL format:
{
  "sample_id": "...",
  "prediction": "...",
  "reference": "..."
}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

from rouge_score import rouge_scorer


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    return " ".join(str(s).strip().split())


def evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1: List[float] = []
    r2: List[float] = []
    rl: List[float] = []

    for r in rows:
        pred = normalize_text(r.get("prediction", ""))
        ref = normalize_text(r.get("reference", ""))
        s = scorer.score(ref, pred)
        r1.append(float(s["rouge1"].fmeasure))
        r2.append(float(s["rouge2"].fmeasure))
        rl.append(float(s["rougeL"].fmeasure))

    def pack(xs: List[float]) -> Dict[str, float]:
        return {"mean": mean(xs) if xs else 0.0, "std": pstdev(xs) if len(xs) > 1 else 0.0}

    return {
        "count": len(rows),
        "rouge1": pack(r1),
        "rouge2": pack(r2),
        "rougeL": pack(rl),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ROUGE from prediction JSONL.")
    ap.add_argument("--pred-jsonl", required=True, help="Input JSONL with sample_id/prediction/reference.")
    ap.add_argument("--out-json", default=None, help="Optional output path for metrics JSON.")
    ap.add_argument("--out-csv", default=None, help="Optional output path for one-row metrics CSV.")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.pred_jsonl))
    metrics = evaluate(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_csv:
        with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["count", "rouge1_mean", "rouge1_std", "rouge2_mean", "rouge2_std", "rougeL_mean", "rougeL_std"])
            w.writerow(
                [
                    metrics["count"],
                    metrics["rouge1"]["mean"],
                    metrics["rouge1"]["std"],
                    metrics["rouge2"]["mean"],
                    metrics["rouge2"]["std"],
                    metrics["rougeL"]["mean"],
                    metrics["rougeL"]["std"],
                ]
            )


if __name__ == "__main__":
    main()
