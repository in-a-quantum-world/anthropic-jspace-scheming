# Plan: 30 hours, six phases, a gate on each

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
