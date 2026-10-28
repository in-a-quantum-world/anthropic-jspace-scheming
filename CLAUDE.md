# Project briefing for Claude Code sessions

Sandbagging in the J-space: can a workspace lens recover an answer a model deliberately withholds, and when recovery fails, is that concealment or measurement failure? Read, in order: README.md (status), docs/plan.md (the current plan, P0-P3), docs/prereg.md (frozen + Amendment 1), docs/week1-findings.md, docs/review-response.md, CLAIMS.md.

## Working rules

- Commits: author "Rucha Agashe" <ruchaagashe212@gmail.com>, no AI attribution lines (her standing instruction).
- Never run any model on data/heldout_bank_v1.json, and never use it in LoRA training, lens fitting, decoder selection, or threshold calibration. It is reserved for the confirmatory run.
- docs/prereg.md original text is frozen; changes only as numbered amendments appended at the end.
- data/matched_distractors.json (v1) is frozen; improvements go to v2 or later files.
- Claims live in CLAIMS.md (append-only); analysis history in docs/analysis-log.md (append-only).
- Every reported table must be reproducible by `python3 src/regenerate_tables.py`; metric definitions live in src/metrics_lib.py and must not be duplicated with different behaviour.
- Prompt strings and read bands: configs/prompts.yaml is the source of truth.
- Writing style for all prose: plain and direct, no em dashes, no AI-isms, British spelling, concrete numbers.

## Local environment (Rucha's Windows machine, RTX 5070 8GB)

- Python venv: C:\Users\rucha\jspace-env (activate before running anything).
- Local fitting: src/fit_lens_5070.py (checkpointed, resumable; keep the machine awake during fits; --dim-batch and --max-seq-len are the tight-VRAM knobs; if a backward pass crawls, set NVIDIA "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" to surface a real OOM).
- Local evaluation: src/run_sandbag_local.py (small models only; 8GB fits up to ~2B for fitting, ~8B for inference).
- In flight as of 2026-10-28: a Qwen/Qwen2.5-1.5B-Instruct lens fit (checkpoint in lens_fits/qwen2.5-1.5b-instruct/), then the sandbag run on it with the fitted lens.
- GPU runs at scale happen in Colab (notebooks/week1_followup.ipynb), not on this machine.

## Current priorities (docs/plan.md has the full table)

P0 remainder: baselines (final-output logits, budgeted black-box elicitation) and scoring-pipeline checks. P1: candidate-recovery evaluation on a validated password-locked organism, and the parent-fitted vs refitted lens matrix. Do not start DeepSeek-V4-Flash work; it is gated and later.
