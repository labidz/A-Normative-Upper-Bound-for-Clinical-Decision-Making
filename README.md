# CDM Guideline-Adherence Study — Zero-Cost Execution Guide

This is the zero-cost variant. Everything runs locally on Colab's free T4 GPU using open-source models. **No API keys. No credit card. No money.**

## The trade-off you're accepting

| | API version | **This version (local)** |
|---|---|---|
| Cost | $15–25 | **$0** |
| Models | GPT-4o-mini, Claude Haiku, Gemini | Phi-3.5 (3.8B), Qwen2.5-7B, OpenBioLLM-8B |
| Speed | 2,400 cases in ~1 day | 600 cases in ~2–3 weeks of evenings |
| Absolute accuracy | Higher | Lower (expected — small models) |
| Paper angle | Benchmarking frontier APIs | **Open-source, privacy-preserving, reproducible** |
| Reviewer reaction | "Did you try larger models?" | "Excellent, we can reproduce this on-premise" |



## Files

| File | Purpose |
|---|---|
| `guidelines.py` | Encodings of the four clinical guidelines |
| `metrics.py` | Four deviation metrics + composite GDI |
| `framework.py` | Local HuggingFace adapter, evidence server, case runner |
| `run_experiment.py` | Main pipeline — cell-by-cell Colab runnable |
| `paper_draft.md` | Paper draft with placeholders |

## Step-by-step execution

### Step 1 — Colab setup (one-time)
Open a new Colab notebook. Change runtime to T4 GPU: `Runtime > Change runtime type > T4 GPU > Save`.

### Step 2 — Upload the code files
Click the folder icon (left sidebar) → upload `guidelines.py`, `metrics.py`, `framework.py`, `run_experiment.py` to `/content/`.

### Step 3 — Mount Drive and put the dataset on it
```python
from google.colab import drive
drive.mount('/content/drive')
```
Upload the MIMIC-IV-Ext CDM v1.1 files from PhysioNet to `/MyDrive/mimic-iv-ext-cdm/1.1/` in your Google Drive.

### Step 4 — Install dependencies
```python
!pip install -q transformers accelerate bitsandbytes scipy scikit-learn \
                statsmodels seaborn tqdm
```

Confirm GPU:
```python
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
```

### Step 5 — Pilot run (do this first — catches schema mismatches cheaply)
Open `run_experiment.py`. In CELL 2, confirm:
```python
N_PER_PATHOLOGY = 25    # 100 cases total — finishes in ~1–2 hours
LOCAL_MODELS_TO_RUN = ["microsoft/Phi-3.5-mini-instruct"]
```

Run cells 2–9 in order, then uncomment and run CELL 10 (the main runner at the end of the file). Phi-3.5 is the lightest (3.8B) and fastest model — if the pipeline works on it, it works everywhere.

If the pilot completes in 1–2 hours with non-empty `trajectories.jsonl` and a sensible metrics table, you're good to scale.

### Step 6 — Full run (one model per Colab session)
Scale up:
```python
N_PER_PATHOLOGY = 150   # 600 cases total
LOCAL_MODELS_TO_RUN = ["microsoft/Phi-3.5-mini-instruct"]
```
Run. Expect ~6–8 hours. The script writes to JSONL as it goes, so if Colab disconnects mid-run just reconnect and rerun CELL 10 — it picks up from where it left off (`resume=True`).

When done, start a **new Colab session**, change `LOCAL_MODELS_TO_RUN` to:
```python
LOCAL_MODELS_TO_RUN = ["Qwen/Qwen2.5-7B-Instruct"]
```
Run again. This will take ~12–16 hours and span 2 sessions. Same JSONL, still resuming.

Finally, a third session with:
```python
LOCAL_MODELS_TO_RUN = ["aaditya/Llama3-OpenBioLLM-8B"]
```
This is the medical-specialized model — critical for the paper's punchline comparison.

### Step 7 — Analysis
Once all three models have finished, run CELL 7 (metrics computation), CELL 8 (statistics), and CELL 9 (figures). Total: a few minutes.

### Step 8 — Fill in the paper draft
Every `[placeholder]` in `paper_draft.md` maps to a specific value in the analysis output (see prior README in this conversation for the full mapping).

## Time budget (be realistic)

| Task | Wall clock |
|---|---|
| Setup + schema debugging | 2–4 hours |
| Pilot run (100 cases × 1 model) | 1–2 hours |
| Full run, Phi-3.5-mini | 6–8 hours (1 session) |
| Full run, Qwen2.5-7B | 12–16 hours (2 sessions) |
| Full run, OpenBioLLM-8B | 14–18 hours (2 sessions) |
| Analysis + figures | <1 hour |
| **Total compute** | **~40–50 hours across ~2 weeks** |
