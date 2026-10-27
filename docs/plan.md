# Plan of record (revised 2026-10-27 after external review)

Central question: when a model withholds an answer, can a workspace lens recover it without knowing the answer in advance, and when recovery fails, is that because of concealment or because the lens no longer measures the model reliably?

Working title: "Can Workspace Lenses Recover Withheld Answers? Separating Trained Concealment from Measurement Failure."

Five outcomes, never conflated: persistence (token appears in readout), candidate recovery (frozen rule picks the answer from a candidate set), open-vocabulary recovery (no gold answer supplied), concealment detection (withholding vs error vs benign non-disclosure), behavioural elicitation (intervention makes the model answer). The operational endpoint for the next experiment is candidate recovery with a frozen decoder; the decoder conditions on the model's observed output and discounts it.

| Priority | Deliverable | Status |
| --- | --- | --- |
| P0 | Validated scoring pipeline, fresh held-out bank, strong baselines (final-output logits, logit lens, J, J++ where compatible, one budgeted black-box elicitation) | Pipeline tests and held-out bank committed 2026-10-27; baselines partly run |
| P1 | Candidate-recovery evaluation on one validated trained organism (password-locked adapter: held-out capability when unlocked, suppressed without trigger, general capability intact, disjoint train/eval facts, multiple seeds preferred) | Not started |
| P1 | Parent-fitted vs refitted lens matrix on the same organism (2 lenses x 4 activation conditions, same fitting corpus and budget, no evaluation answers in fitting; honest-finetuning control if feasible) | Fitting pipeline validated on deepseek-coder-1.3b 2026-10-27 |
| P2 | One targeted causal test: does a specific representation suppress an otherwise recoverable answer, with layer- and norm-matched random controls, an unrelated direction, and effects on both readout and generations | Not started |
| P2 | External organism or second-family replication | Not started |
| P3 | Frozen DeepSeek-V4-Flash replication (gates: exact revisions and hook compatibility; positive controls; frozen small benchmark; runtime and peak-memory measurement; logit baselines; no tuning on the final bank; note the four-parallel-residual-stream mean convention) | Later |
| Later | Agentic monitoring and jailbreak-defence evaluation (different threat model and benchmark) | Out of current scope |

Next 30 human hours: P0 plus one well-validated P1 result. Not all rows.

The original six-phase plan below is retained for the record; its gates were all passed or honestly failed and logged (see docs/analysis-log.md).

| Phase | Hours | Output | Gate |
| --- | --- | --- | --- |
| Feasibility | 0-3 | Lens loads; positive control; 20-pair sandbag pilot | Lens can't see known answers: switch model (12B) or revert per plan |
| Behaviour | 4-8 | ~150-question filtered bank; compliance rates with CIs | Compliance under 50%: strengthen instruction wording |
| Readout | 9-14 | Layer-by-layer J-space contents, both conditions, 2 models | Noisy readouts: restrict to top-k concepts |
| Analysis | 15-20 | Persistence and recovery rates across conditions | No signal: report structural contrast with coverage caveat |
| Intervention | 21-26 | Ablation effects vs norm-matched baselines; stretch: password-locked LoRA arm | A null is reported, not rescued |
| Write-up | 27-30 | LessWrong post, this repo finalised, readout figures | Scope frozen; nothing new starts |

Funding: BlueDot rapid grant, staged ($100 tranche 1: GPU for pilot + readouts + API credits for dataset work; tranche 2 after gates pass: LoRA arm and lens-fitting contingency).

## Working agreement (solo + AI tooling)

- Research decisions, gate verdicts, transcript eyeballing, claims: human only.
- Desk work, drafting, analysis design, next-stage scripts: Claude (project chat).
- On-machine implementation and debugging: coding agent on the GPU box with narrow briefs and a definition of done per task; unattended sweeps only after the pipeline is validated, committing to `run/*` branches.
- OpenAI API credits: dataset generation and grading checks only; the subject models are open-weight and run locally.
