# WindyBench: Wind-Agent Benchmark

Open-source benchmark project for multi-agent wind farm O&M diagnosis with RAG-based reasoning.

## Overview

This project implements a paper-aligned multi-agent pipeline:
- `SPA`: Signal Perception Agent
- `TKBA/TKRA`: Technical Knowledge Base + Retrieval Augmentation
- `CRA`: Contextual Refinement Agent
- `ERA`: Expert Reasoning Agent

## Project Structure

```text
.
├── data/
│   └── README.md
├── scripts/
│   ├── s01_generate_qa_dataset.py
│   ├── s02_cpt_domain_pretrain.py
│   ├── s03_build_cot_sft_dataset.py
│   ├── s04_finetune_lora_llama7b.py
│   ├── s05_build_tkba_index.py
│   ├── s06_benchmark_tkra.py
│   ├── s07_bdma_multi_agent_infer.py
│   └── s08_evaluate_rouge.py
├── requirements.txt
└── README.md
```

## Dataset

Local dataset files are intentionally removed from this repository.

Download the dataset from Google Drive:
- https://drive.google.com/drive/folders/1jFUpqZkpE0wMmp8MievHeYyjzy94CObM?usp=drive_link

After downloading, place required files in the repository root (or adjust script paths).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Benchmark Pipeline

### Stage 1: Build QA dataset

```bash
python scripts/s01_generate_qa_dataset.py \
  --turbine-json turbine_80.json \
  --plant-json wind_plant_data.json \
  --output-jsonl qa_dataset_debug.jsonl \
  --num-samples 968
```

### Stage 2: CPT (optional)

```bash
python scripts/s02_cpt_domain_pretrain.py \
  --model-name-or-path <base_model> \
  --corpus <domain_corpus_files...> \
  --output-dir outputs/cpt_adapter
```

### Stage 3: Build CoT-SFT dataset

```bash
python scripts/s03_build_cot_sft_dataset.py \
  --qa-jsonl qa_dataset_debug.jsonl \
  --out-jsonl cot_sft_dataset.jsonl \
  --teacher-model llama2:7b
```

### Stage 4: LoRA/QLoRA SFT

```bash
python scripts/s04_finetune_lora_llama7b.py \
  --model-name-or-path <base_model> \
  --train-jsonl cot_sft_dataset.jsonl \
  --dataset-format cot \
  --output-dir outputs/llama7b_lora \
  --use-4bit \
  --auto-split
```

### Stage 5: Build TKBA index

```bash
python scripts/s05_build_tkba_index.py build \
  --qa-jsonl qa_dataset_debug.jsonl \
  --out-dir rag_kb
```

### Stage 6: Run benchmark

```bash
python scripts/s06_benchmark_tkra.py \
  --test-jsonl qa_dataset_debug.jsonl \
  --out-dir benchmark_out \
  --kb-dir rag_kb
```

### Stage 7: Multi-agent inference

```bash
python scripts/s07_bdma_multi_agent_infer.py \
  --kb-dir rag_kb \
  --manifest-inline '{"component_id":"Transmission / Main Bearing","alarm_desc":"M.bear. Error Pressure","mean_abs_z_primary":4.9,"trend_description":"Rapidly Increasing","operating_state":"High wind + rated power"}' \
  --llm-model llama2:7b \
  --top-k 5 \
  --use-rerank \
  --hyde
```

### Stage 8: ROUGE evaluation

```bash
python scripts/s08_evaluate_rouge.py --pred-jsonl benchmark_out/rag_hyde_rerank.pred.jsonl
```

## Notes

- Default local backend is Ollama.
- Retrieval and generation settings should match your paper protocol for final reporting.
