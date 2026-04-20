# CDM Guideline-Adherence Study — Zero-Cost Execution Guide

This is the zero-cost variant. Everything runs locally on Colab's free T4 GPU using open-source models. **No API keys. No cost.**

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

