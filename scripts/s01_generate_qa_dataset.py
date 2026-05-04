#!/usr/bin/env python3
"""
Generate wind-turbine QA fine-tuning samples from raw JSON data.

Data source:
- turbine_80.json: dense SCADA time series + alarm intervals
- wind_plant_data.json: alarm dictionary metadata

Output:
- JSONL, each line contains:
  - raw IDs/values/time-series slices from source data
  - z-score based anomaly evidence
  - LLM-generated question + fixed-format answer:
      Cause Analysis / Action Recommendations / Risk Evaluation
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EXCLUDE_ANALOG_KEYS = {"turbine_id", "date_time"}
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
AutoTokenizer = None
AutoModelForCausalLM = None
torch = None
_HF_MODEL_CACHE: Dict[str, Any] = {}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_alarm_lookup(alarm_dictionary: Dict[str, List[Any]]) -> Dict[int, Dict[str, Any]]:
    keys = list(alarm_dictionary.keys())
    if not keys:
        return {}

    n = len(alarm_dictionary[keys[0]])
    lookup: Dict[int, Dict[str, Any]] = {}
    for i in range(n):
        row = {k: alarm_dictionary[k][i] for k in keys}
        aid = int(row.get("alarm_id", -1))
        lookup[aid] = row
    return lookup


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        return None
    return None


def summarize_trend(values: List[float]) -> str:
    if len(values) < 4:
        return "Insufficient trend data"
    n = len(values)
    a = statistics.mean(values[: max(1, n // 3)])
    b = statistics.mean(values[-max(1, n // 3) :])
    if b - a > 0.25 * (abs(a) + 1e-6):
        return "Increasing"
    if a - b > 0.25 * (abs(a) + 1e-6):
        return "Decreasing"
    return "Stable to slowly drifting"


def z_severity(mean_abs_z: float) -> str:
    if mean_abs_z >= 5.0:
        return "Critical"
    if mean_abs_z >= 3.0:
        return "Moderate"
    return "Mild"


def pick_keywords(alarm_text: str) -> List[str]:
    text = alarm_text.lower()
    mapping = [
        (["temp", "temperature", "overheat"], ["tmp", "temp", "bearing", "oil"]),
        (["vibration", "vib"], ["vib", "acc", "bearing"]),
        (["pressure", "hydraulic", "hyd"], ["pres", "hyd", "act", "pitch", "oil"]),
        (["pitch", "blade"], ["pitch", "blade", "hyd"]),
        (["yaw"], ["yaw", "azimuth"]),
        (["generator", "grid", "voltage", "current", "power"], ["gri", "phv", "a_phs", "pwr", "hz", "gen"]),
        (["speed", "rpm", "rotor"], ["rpm", "speed", "rot", "gen"]),
        (["brake"], ["brake", "hyd", "pres"]),
    ]
    out: List[str] = []
    for words, kws in mapping:
        if any(w in text for w in words):
            out.extend(kws)
    if not out:
        out = ["tmp", "pwr", "pitch", "pres", "rpm", "vib", "hz"]
    return sorted(set(out))


def choose_candidate_sensors(sensor_names: List[str], alarm_text: str, max_candidates: int) -> List[str]:
    keywords = pick_keywords(alarm_text)
    scored: List[Tuple[int, str]] = []
    for s in sensor_names:
        low = s.lower()
        score = sum(1 for kw in keywords if kw in low)
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [s for _, s in scored[:max_candidates]]
    if selected:
        return selected
    return sensor_names[:max_candidates]


def compact_series(
    timestamps: List[str], values: List[float], zscores: List[float], max_points: int
) -> List[Dict[str, Any]]:
    if not values:
        return []
    if len(values) <= max_points:
        idxs = list(range(len(values)))
    else:
        step = max(1, len(values) // max_points)
        idxs = list(range(0, len(values), step))[:max_points]
    return [
        {"ts": timestamps[i], "value": round(values[i], 6), "z": round(zscores[i], 6)}
        for i in idxs
    ]


def get_operating_state(analog_data: Dict[str, List[Any]], idx: int) -> str:
    sensor_names = [k for k in analog_data.keys() if k not in EXCLUDE_ANALOG_KEYS]

    def find_value(preferred: List[str], fallback_keywords: List[str]) -> Optional[Tuple[str, float]]:
        for name in preferred:
            if name in analog_data:
                v = safe_float(analog_data[name][idx])
                if v is not None:
                    return name, v
        for s in sensor_names:
            low = s.lower()
            if any(kw in low for kw in fallback_keywords):
                v = safe_float(analog_data[s][idx])
                if v is not None:
                    return s, v
        return None

    wind = find_value(
        preferred=["wnac_avg_WSpd1", "wnac_avg_WSpd2"],
        fallback_keywords=["wspd", "wind"],
    )
    power = find_value(
        preferred=["wgdc_avg_TriGri_PwrAt"],
        fallback_keywords=["pwrat", "active", "pwrat"],
    )
    rotor = find_value(
        preferred=["wgen_avg_Spd", "wgen_avg_RtrSpd_IGR", "wgen_avg_RtrSpd_WP2035"],
        fallback_keywords=["rtrspd", "gen_avg_spd", "_spd"],
    )

    parts: List[str] = []
    if wind:
        parts.append(f"wind speed={wind[1]:.3f} ({wind[0]})")
    if power:
        parts.append(f"active power={power[1]:.3f} ({power[0]})")
    if rotor:
        parts.append(f"rotor/generator speed={rotor[1]:.3f} ({rotor[0]})")

    if not parts:
        return "Operating state not available from selected channels"
    return "; ".join(parts)


def make_prompt(sample: Dict[str, Any]) -> str:
    manifest = sample["manifest"]
    primary = sample["evidence"][0] if sample["evidence"] else {}
    primary_preview = {
        "sensor_id": primary.get("sensor_id"),
        "mean_abs_z": primary.get("mean_abs_z"),
        "trend": primary.get("trend"),
        "event_min": primary.get("event_min"),
        "event_max": primary.get("event_max"),
        "series_points": primary.get("series_points", [])[:8],
    }
    q_modes = [
        "Root-cause oriented: ask for the most likely fault mechanism from evidence.",
        "Maintenance planning: ask for prioritized maintenance actions and checks.",
        "Risk-centric: ask for short-term operational risk if no intervention is applied.",
        "Verification-centric: ask how to confirm diagnosis using additional SCADA checks.",
    ]
    q_mode = random.choice(q_modes)
    evidence_preview = [
        {
            "sensor_id": e["sensor_id"],
            "severity_score_mean_abs_z": e["mean_abs_z"],
            "trend": e["trend"],
            "series_points": e["series_points"][:8],
        }
        for e in sample["evidence"][:3]
    ]

    return f"""
You are an expert wind-turbine diagnostic engineer.
Create ONE QA training sample from the provided manifest and SCADA evidence.

Hard constraints:
1) Return ONLY valid JSON, no markdown, no extra text.
2) JSON schema:
{{
  "question": "<one question>",
  "answer": "Cause Analysis:\\n...\\n\\nAction Recommendations:\\n...\\n\\nRisk Evaluation:\\n..."
}}
3) Question must include at least one concrete sensor_id and at least one numeric value or z-score.
4) Use this question style target: {q_mode}
5) Answer must be concise, technical, and actionable; do not invent fields.

Input Manifest:
- Turbine ID: {sample["turbine_id"]}
- Alarm ID: {sample["alarm_id"]}
- Alarm Description: {sample["alarm_desc"]}
- Component ID: {manifest["component_id"]}
- Event Start: {sample["event_start"]}
- Event End: {sample["event_end"]}
- Severity Level: {manifest["severity_level"]}
- Mean Abs Z (Primary): {manifest["mean_abs_z_primary"]}
- Trend Description: {manifest["trend_description"]}
- Operating State: {manifest["operating_state"]}

Primary Evidence:
{json.dumps(primary_preview, ensure_ascii=False)}

Additional Evidence (Top-3):
{json.dumps(evidence_preview, ensure_ascii=False)}
""".strip()


def parse_qa_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    def _answer_obj_to_text(a_obj: Any) -> Optional[str]:
        if isinstance(a_obj, str):
            return a_obj.strip()
        if isinstance(a_obj, dict):
            cause = str(a_obj.get("Cause Analysis", "")).strip()
            action = str(a_obj.get("Action Recommendations", "")).strip()
            risk = str(a_obj.get("Risk Evaluation", "")).strip()
            if cause or action or risk:
                return (
                    "Cause Analysis:\n"
                    + cause
                    + "\n\nAction Recommendations:\n"
                    + action
                    + "\n\nRisk Evaluation:\n"
                    + risk
                ).strip()
        return None

    text = text.strip()
    # Try strict JSON parse first.
    try:
        obj = json.loads(text)
        q = obj.get("question")
        a = _answer_obj_to_text(obj.get("answer"))
        if isinstance(q, str) and isinstance(a, str):
            return q.strip(), a
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object block.
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            q = obj.get("question")
            a = _answer_obj_to_text(obj.get("answer"))
            if isinstance(q, str) and isinstance(a, str):
                return q.strip(), a
        except json.JSONDecodeError:
            pass

    # Fallback: if model outputs free text with explicit section headings.
    if all(k in text for k in ["Cause Analysis:", "Action Recommendations:", "Risk Evaluation:"]):
        # Try recovering a question if present, otherwise keep None and caller can fallback.
        qm = re.search(
            r"(?:^|\n)\s*Question\s*[:：]\s*(.*?)\s*(?:\n\s*Answer\s*[:：]|\n\s*Cause Analysis:)",
            text,
            flags=re.S | re.I,
        )
        q = qm.group(1).strip() if qm else None

        am = re.search(r"(Cause Analysis:.*)", text, flags=re.S)
        a = am.group(1).strip() if am else None
        if isinstance(a, str) and a:
            return q, a

    return None, None


def ensure_answer_sections(answer: str, sample: Dict[str, Any]) -> str:
    required = ["Cause Analysis:", "Action Recommendations:", "Risk Evaluation:"]
    if all(r in answer for r in required):
        return answer

    # Fallback structure if model drifts from required format.
    primary = sample["evidence"][0] if sample["evidence"] else {}
    cause = (
        f"The alarm {sample['alarm_id']} ({sample['alarm_desc']}) is associated with "
        f"{sample['manifest']['component_id']}. The primary channel "
        f"{primary.get('sensor_id', 'N/A')} shows elevated anomaly intensity "
        f"(mean|z|={primary.get('mean_abs_z', 'N/A')})."
    )
    action = (
        "Inspect the affected subsystem, verify sensor integrity, perform targeted maintenance "
        "following OEM procedures, and validate recovery by trending z-score back toward baseline."
    )
    risk = (
        "If unresolved, fault progression may increase downtime risk and secondary component stress. "
        "Escalate to controlled shutdown when anomaly severity remains critical."
    )
    return (
        "Cause Analysis:\n"
        + cause
        + "\n\nAction Recommendations:\n"
        + action
        + "\n\nRisk Evaluation:\n"
        + risk
    )


def call_ollama_generate(model: str, prompt: str, timeout_sec: int = 180) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return obj.get("response", "").strip()


def parse_model_spec(spec: str) -> Dict[str, str]:
    # Supported:
    # - ollama:llama2:7b
    # - hf:/data_hdd/.../Llama-2-7b-hf
    # Backward-compatible fallback: plain string -> ollama model name
    if spec.startswith("ollama:"):
        return {"backend": "ollama", "model": spec[len("ollama:") :]}
    if spec.startswith("hf:"):
        return {"backend": "hf", "model": spec[len("hf:") :]}
    return {"backend": "ollama", "model": spec}


def _lazy_import_transformers_torch() -> None:
    global AutoTokenizer, AutoModelForCausalLM, torch
    if AutoTokenizer is not None and AutoModelForCausalLM is not None and torch is not None:
        return
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM, AutoTokenizer as _AutoTokenizer

    torch = _torch
    AutoTokenizer = _AutoTokenizer
    AutoModelForCausalLM = _AutoModelForCausalLM


def _get_hf_model(model_path: str):
    if model_path in _HF_MODEL_CACHE:
        return _HF_MODEL_CACHE[model_path]
    _lazy_import_transformers_torch()
    local_only = Path(model_path).exists()
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
    _HF_MODEL_CACHE[model_path] = (tokenizer, model)
    return tokenizer, model


def call_hf_local(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 320,
    temperature: float = 0.3,
) -> str:
    tokenizer, model = _get_hf_model(model_path)
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


def call_llm_generate(model_spec: str, prompt: str, timeout_sec: int = 180) -> str:
    info = parse_model_spec(model_spec)
    if info["backend"] == "ollama":
        return call_ollama_generate(info["model"], prompt, timeout_sec=timeout_sec)
    if info["backend"] == "hf":
        return call_hf_local(info["model"], prompt, max_new_tokens=320, temperature=0.3)
    raise ValueError(f"Unknown model backend in spec: {model_spec}")


def build_anomaly_evidence(
    analog_data: Dict[str, List[Any]],
    idx_start: int,
    idx_end: int,
    baseline_points: int,
    max_candidates: int,
    top_k_sensors: int,
    max_series_points: int,
    alarm_text: str,
) -> List[Dict[str, Any]]:
    date_time = analog_data["date_time"]
    sensor_names = [k for k in analog_data.keys() if k not in EXCLUDE_ANALOG_KEYS]
    candidates = choose_candidate_sensors(sensor_names, alarm_text, max_candidates=max_candidates)

    b_start = max(0, idx_start - baseline_points)
    b_end = idx_start
    if b_end - b_start < 5:
        return []

    evidence: List[Dict[str, Any]] = []
    ts_slice = date_time[idx_start : idx_end + 1]

    for sensor in candidates:
        event_raw = analog_data[sensor][idx_start : idx_end + 1]
        base_raw = analog_data[sensor][b_start:b_end]

        event_vals = [safe_float(v) for v in event_raw]
        base_vals = [safe_float(v) for v in base_raw]
        event_vals_num = [v for v in event_vals if v is not None]
        base_vals_num = [v for v in base_vals if v is not None]
        if len(event_vals_num) < 5 or len(base_vals_num) < 20:
            continue

        mu = statistics.mean(base_vals_num)
        sigma = statistics.pstdev(base_vals_num)
        if sigma <= 1e-9:
            continue

        zscores: List[float] = []
        vals_aligned: List[float] = []
        ts_aligned: List[str] = []
        for i, v in enumerate(event_vals):
            if v is None:
                continue
            z = (v - mu) / sigma
            vals_aligned.append(v)
            zscores.append(z)
            ts_aligned.append(ts_slice[i])

        if len(vals_aligned) < 5:
            continue

        mean_abs_z = statistics.mean(abs(z) for z in zscores)
        trend = summarize_trend(vals_aligned)
        evidence.append(
            {
                "sensor_id": sensor,
                "baseline_mean": round(mu, 6),
                "baseline_std": round(sigma, 6),
                "event_mean": round(statistics.mean(vals_aligned), 6),
                "event_min": round(min(vals_aligned), 6),
                "event_max": round(max(vals_aligned), 6),
                "mean_abs_z": round(mean_abs_z, 6),
                "trend": trend,
                "series_points": compact_series(ts_aligned, vals_aligned, zscores, max_series_points),
            }
        )

    evidence.sort(key=lambda x: x["mean_abs_z"], reverse=True)
    return evidence[:top_k_sensors]


def make_question_fallback(sample: Dict[str, Any]) -> str:
    primary = sample["evidence"][0] if sample["evidence"] else {}
    sid = primary.get("sensor_id", "unknown_sensor")
    z = primary.get("mean_abs_z", "N/A")
    pts = primary.get("series_points", [])
    if pts:
        v0 = pts[0].get("value", "N/A")
        v1 = pts[-1].get("value", "N/A")
    else:
        v0 = "N/A"
        v1 = "N/A"

    templates = [
        (
            f"For turbine {sample['turbine_id']}, alarm {sample['alarm_id']} ({sample['alarm_desc']}) "
            f"occurred from {sample['event_start']} to {sample['event_end']}. "
            f"Given sensor {sid} with mean|z|={z} and event value drift {v0}->{v1}, "
            "what is the most likely root cause, what maintenance actions should be prioritized, "
            "and what operational risks should be reported?"
        ),
        (
            f"During alarm {sample['alarm_id']} on turbine {sample['turbine_id']}, "
            f"{sid} showed abnormal behavior (mean|z|={z}) in the interval {sample['event_start']} ~ {sample['event_end']}. "
            "Please diagnose the probable failure mechanism, provide a stepwise maintenance strategy, "
            "and evaluate the risk if this anomaly continues."
        ),
        (
            f"Using the SCADA anomaly evidence for {sample['manifest']['component_id']} "
            f"(primary sensor: {sid}, mean|z|={z}, values {v0}->{v1}), "
            f"how should we explain alarm {sample['alarm_id']} ({sample['alarm_desc']}) "
            "in terms of cause, recommended intervention, and residual risk?"
        ),
    ]
    return random.choice(templates)


def generate_dataset(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    turbine = load_json(Path(args.turbine_json))
    plant = load_json(Path(args.plant_json))

    analog_data: Dict[str, List[Any]] = turbine["analog_data"]
    alarms: Dict[str, List[Any]] = turbine["alarms"]
    alarm_lookup = build_alarm_lookup(plant["alarm_dictionary"])
    dt = analog_data["date_time"]

    n_alarms = len(alarms["alarm_id"])
    indices = list(range(n_alarms))
    random.shuffle(indices)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for idx in indices:
            if generated >= args.num_samples:
                break

            alarm_id = int(alarms["alarm_id"][idx])
            alarm_desc = str(alarms["alarm_desc"][idx])
            if args.skip_system_ok and alarm_id == 0:
                continue

            event_start = str(alarms["date_time_ini"][idx])
            event_end = str(alarms["date_time_end"][idx])
            s = bisect.bisect_left(dt, event_start)
            e = bisect.bisect_right(dt, event_end) - 1
            if s < 0 or e <= s or e >= len(dt):
                skipped += 1
                continue

            md = alarm_lookup.get(alarm_id, {})
            component_id = (
                f"{md.get('alarm_system', 'UnknownSystem')} / "
                f"{md.get('alarm_subsystem', 'UnknownSubsystem')}"
            )

            evidence = build_anomaly_evidence(
                analog_data=analog_data,
                idx_start=s,
                idx_end=e,
                baseline_points=args.baseline_points,
                max_candidates=args.max_candidate_sensors,
                top_k_sensors=args.top_k_sensors,
                max_series_points=args.max_series_points,
                alarm_text=f"{alarm_desc} {md.get('alarm_system', '')} {md.get('alarm_subsystem', '')}",
            )
            if not evidence:
                skipped += 1
                continue

            primary = evidence[0]
            manifest = {
                "component_id": component_id,
                "severity_level": z_severity(primary["mean_abs_z"]),
                "mean_abs_z_primary": primary["mean_abs_z"],
                "trend_description": primary["trend"],
                "operating_state": get_operating_state(analog_data, s),
            }

            sample: Dict[str, Any] = {
                "sample_id": f"t{turbine['turbine_id']}_a{alarm_id}_i{idx}",
                "turbine_id": turbine["turbine_id"],
                "alarm_id": alarm_id,
                "alarm_desc": alarm_desc,
                "alarm_system": md.get("alarm_system"),
                "alarm_subsystem": md.get("alarm_subsystem"),
                "event_start": event_start,
                "event_end": event_end,
                "event_index_range": {"start_idx": s, "end_idx": e},
                "manifest": manifest,
                "evidence": evidence,
            }

            question: Optional[str] = None
            answer: Optional[str] = None
            if not args.no_llm:
                prompt = make_prompt(sample)
                for _ in range(args.llm_retries):
                    try:
                        llm_spec = args.llm_model if args.llm_model else f"ollama:{args.ollama_model}"
                        raw = call_llm_generate(llm_spec, prompt, timeout_sec=args.llm_timeout_sec)
                        q, a = parse_qa_from_text(raw)
                        if q and a:
                            question, answer = q, ensure_answer_sections(a, sample)
                            break
                    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                        time.sleep(1.0)

            if question is None or answer is None:
                question = make_question_fallback(sample)
                answer = ensure_answer_sections("", sample)

            sample["question"] = question
            sample["answer"] = answer
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            generated += 1

            if generated % 10 == 0:
                print(f"generated={generated}, skipped={skipped}")

    print(f"Done. generated={generated}, skipped={skipped}, output={out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate QA dataset for wind-turbine diagnosis experiments.")
    p.add_argument("--turbine-json", default="turbine_80.json", help="Path to turbine_80 style JSON")
    p.add_argument("--plant-json", default="wind_plant_data.json", help="Path to wind_plant_data.json")
    p.add_argument("--output-jsonl", default="qa_dataset_turbine80.jsonl", help="Output JSONL path")
    p.add_argument("--num-samples", type=int, default=100, help="How many QA samples to generate")
    p.add_argument(
        "--baseline-points",
        type=int,
        default=288,
        help="Baseline window size before event (points). 288 ~= 24h for 5-min data",
    )
    p.add_argument("--top-k-sensors", type=int, default=5, help="Top-K anomaly sensors to keep")
    p.add_argument("--max-candidate-sensors", type=int, default=40, help="Max sensors scored per alarm")
    p.add_argument("--max-series-points", type=int, default=48, help="Max stored points per sensor series")
    p.add_argument(
        "--llm-model",
        default=None,
        help="Model spec: ollama:<name> or hf:</local/path>. If unset, falls back to --ollama-model.",
    )
    p.add_argument("--ollama-model", default="llama2:7b", help="Ollama model name")
    p.add_argument("--llm-timeout-sec", type=int, default=180, help="Per-call LLM timeout seconds")
    p.add_argument("--llm-retries", type=int, default=2, help="Retries per sample when LLM output is invalid")
    p.add_argument("--no-llm", action="store_true", help="Skip Ollama call and use fallback question/answer")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--skip-system-ok", action="store_true", default=True, help="Skip alarm_id=0")
    p.add_argument("--include-system-ok", action="store_true", help="Include alarm_id=0 records")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.include_system_ok:
        args.skip_system_ok = False
    generate_dataset(args)
