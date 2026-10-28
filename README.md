# Sandbagging in the J-space

When a model withholds an answer, can a workspace lens recover it without knowing the answer in advance, and when recovery fails, is that because of concealment or because the lens no longer measures the model reliably?

## Status (2026-10-27)

- **Completed, exploratory:** a three-model pilot and a four-model matched-distractor run with full per-layer trajectories (`results/`), a read-depth sweep on 12B, a five-readout comparison on Qwen3.6-27B, and a read-only comparison of Dolos-1.5B against its parent. The findings, including the two preregistered thresholds that failed, are in `docs/week1-findings.md`. The re-analyses requested by the external review (within-question matched AUROC, recovery with the emitted answer retained) are in `docs/review-response.md`.
- **Registered:** `docs/prereg.md` (original text frozen; Amendment 1 appended 2026-10-27).
- **Planned:** a fresh held-out question bank with audited distractors, a comparison of parent-fitted and refitted lenses on a password-locked fine-tune of the same model, one targeted causal test, and a DeepSeek-V4-Flash replication with the analysis fixed in advance.
- Reproducibility: `python3 src/regenerate_tables.py` rebuilds every reported table from `results/`. The exact prompt strings and read bands are in `configs/prompts.yaml`, the claims are listed with their limitations in `CLAIMS.md`, and the project history is in `docs/analysis-log.md`.
- Current summary, at pilot strength: there is exploratory evidence that correct-answer tokens remain distinguishable from matched distractors during instructed underperformance, and that visibility depends strongly on read depth. The wrong answer the model is about to give dominates the readout, so a practical recovery method has to discount the model's observed output.

## The question

Sandbagging, a model hiding what it knows during an evaluation, undermines the validity of capability and safety evaluations. This project uses the Jacobian lens from Anthropic's J-space work to test whether sandbagged knowledge remains visible internally:

1. **Persistence (RQ1):** does the correct answer stay elevated in the workspace while the output is deliberately wrong?
2. **Recovery (RQ2):** what fraction of withheld answers can the lens read out, against norm-matched baselines?
3. **Causation (RQ3):** does ablating the concealment-related concepts restore honest answering?

A robustness arm uses a password-locked LoRA, in which the sandbagging is trained into the weights rather than prompted, to check that the monitor is not simply reading the instruction text.

## Design

- Open-ended factual questions with single-token answers; the correct token never appears in the prompt, so anything found internally is model-generated rather than prompt echo.
- Honest and sandbagged conditions on the same filtered question bank (hand-written items plus filtered TriviaQA items from the [solarkyle/jspace-lenses](https://huggingface.co/solarkyle/jspace-lenses) traces).
- Positive controls on every run: the lens must see answers the model gets right honestly, or the run is void.
- Confidence intervals on every proportion, an analysis plan frozen before interventions run, and null results reported as results.

## Models and tooling

| Model | Lens | Role |
| --- | --- | --- |
| google/gemma-4-E4B-it (4B) | solarkyle/jspace-lenses | pilot, free tier |
| google/gemma-4-12B-it | solarkyle/jspace-lenses | primary |
| Qwen/Qwen3.6-27B | solarkyle/jspace-lenses | cross-family check |

Lens library: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache 2.0). Lenses fitted in bf16 read 4-bit activations, so everything here runs on a single GPU.

## Gates

| Phase | Gate |
| --- | --- |
| Feasibility | lens loads; positive control: answers the model knows are visible in the workspace |
| Behaviour | sandbag compliance at a usable rate, else strengthen wording or escalate model |
| Readout | readable layer-by-layer traces, else restrict to top-k |
| Analysis | persistence or structural contrast, reported with lens-coverage caveats |
| Intervention | effects vs norm-matched baselines; a null is a result |
| Write-up | scope frozen |

## Repository layout

- `notebooks/`: the runnable pipeline notebooks
- `results/`: one folder per run (`run/YYYYMMDD-model` branches merge here); raw JSON plus a short notes file
- `docs/`: plan, preregistration, findings, review response, reference pack, lens cards and logs
- `src/`: analysis and fitting scripts; shared metric definitions live in `src/metrics_lib.py`
