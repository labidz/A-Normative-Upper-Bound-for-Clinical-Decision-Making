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
| Writing (filling placeholders + discussion) | 1–2 weeks |
| Physician review + revision | 1–2 weeks |
| **Realistic submission timeline** | **5–7 weeks from today** |

Colab free tier typically gives ~3–4 hour sessions before disconnecting for idleness and ~12 hours maximum. The resume logic handles disconnects gracefully.

**Alternative: Kaggle notebooks.** Kaggle gives 30 GPU-hours/week on T4 x2 or P100. If you're hitting Colab session limits, Kaggle is a better venue for this workload — same code, different platform.

## The five most likely things to break

1. **Column name mismatches on first load.** MIMIC-IV-Ext CDM CSVs may use slightly different column names than I coded against. Symptom: `KeyError: 'valuestr'` or similar. Fix: in a Colab cell, `print(data['labs'].columns.tolist())`, then edit `framework.py::EvidenceServer.serve()` to match. Budget: 30 min.

2. **HuggingFace gated-model errors.** OpenBioLLM is ungated. If you choose a gated model (some Llama variants), you need to (a) accept the license on huggingface.co, (b) create a free HF token, (c) run `from huggingface_hub import login; login()` in Colab. I picked ungated models throughout to avoid this.

3. **OOM on 7B/8B models.** Symptom: `CUDA out of memory`. Fix: ensure no other model is loaded (`del` the previous adapter and run `torch.cuda.empty_cache()`). The `HuggingFaceAdapter.unload()` method exists for this. If still OOM, switch to `Qwen/Qwen2.5-3B-Instruct` (smaller) or reduce `max_new_tokens` in the adapter from 400 to 250.

4. **Models that ignore the tag format.** Small models sometimes output text without `<request>` or `<diagnosis>` tags. The framework will nudge once but then move on. If the nudge rate is high (>20% of turns), increase the explicit examples in `framework.py::SYSTEM_PROMPT`. Phi-3.5 and Qwen2.5 are both reliable; OpenBioLLM is the one most likely to need prompt tuning.

5. **Drive remount required mid-run.** Colab silently unmounts Drive after a while. Fix: remount and rerun CELL 10 (it resumes).

## On the paper framing (important)

Because you are benchmarking **open-source** models, reframe the paper's narrative in your final draft:

- **Title tweak:** "Guideline-Adherence Quantification and Deviation Taxonomy for **Open-Source** Large Language Models on Real-World Electronic Health Record Data"
- **Abstract:** add "All evaluated models are open-weights, enabling on-premise deployment in privacy-preserving clinical settings where API-based frontier models are not viable."
- **Discussion:** add a subsection titled "Deployment implications for privacy-preserving clinical AI" arguing that open-source models are the only legally-viable option in many EU/US clinical contexts (HIPAA, GDPR), making their guideline adherence properties more consequential, not less.
- **Model selection justification:** the three-model set (Phi-3.5 fast general / Qwen2.5 strong general / OpenBioLLM medical-specialized) gives the key contrast reviewers want: does medical pretraining improve guideline adherence, or only accuracy? That's a genuinely interesting question that falls out of your design.

This is not spin. It is an accurate reframing of a paper that now has a different clinical significance. A reviewer who would otherwise write "why didn't you test GPT-4?" instead writes "important contribution for deployable clinical AI."

## What you absolutely must still do

1. **Find a physician co-author.** Still the single biggest acceptance-probability lever. Post on medtwitter, ask at your local med school, email authors of the Hager paper. You need someone board-certified who will sign off on the guideline encodings.

2. **Validate the guideline encodings.** Read the primary sources cited in `guidelines.py` and walk through each encoding function. I encoded from training knowledge; reviewers will probe.

3. **Pre-register the analysis plan** on OSF before running the full experiment. Free, takes an hour, and pre-empts the "you p-hacked" accusation.

## Realistic final expectations

Acceptance probability at AIIM first submission with this execution, clean data, and a physician co-author: ~35–45%. Without a physician co-author: ~20–25%. One round of revisions at a second journal is the expected path.

Total time from today to paper-in-press: 5–9 months. The experiment is 2–3 weeks of it; most of it is writing, review, revision.
