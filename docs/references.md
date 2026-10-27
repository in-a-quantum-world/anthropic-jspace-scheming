# Reference pack: everything known in the J-space ecosystem

For manual review. Checkbox = opened and verified by me (Rucha). Items marked (verify) have details recalled by Claude from training rather than fetched pages. Collated 2026-10-25.

## Core method

- [ ] Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models* — https://transformer-circuits.pub/2026/workspace/index.html — the J-space paper. Author list inconsistent across community citations (Gurnee et al. vs Lindsey et al.): settle it from the page.
- [ ] anthropics/jacobian-lens (GitHub) — reference implementation, Apache 2.0, includes the fitting script. The `jlens` library the pilot uses.
- [ ] neuronpedia/jacobian-lens (HF collection) — Neuronpedia's lens fits, includes a Pythia-70m lens used by others as a validation target.
- [ ] Lens lineage, for the paper's related work: logit lens (nostalgebraist, LessWrong 2020), tuned lens (Belrose et al., arXiv 2303.08112, verify).
- [x] **koayon/jpp-lenses** — https://hf.co/koayon/jpp-lenses — J++ Lens fits (Ayonrinde & Lindsey, LessWrong 2026: "J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses"). Covers **Qwen/Qwen3.6-27B** (our checkpoint; recall@10 55.2%, layers 0-62, 6.6 GB fp32), plus Qwen3.5-9B, gemma-4-31B, Olmo-3-32B, Qwen3.5-122B-A10B, DeepSeek-V4-Flash. File format documented (torch dict, `parameters["jacobians"]`); readout = unembed(J_l h). Code: github.com/safety-research/jpp_lens (current address; the older koayon/jpp_lens does not resolve). Verified from the repo README 2026-10-25; our lens-quality control arm.

## Lens fits and tools (Hugging Face)

- [ ] **solarkyle/jspace-lenses** — https://hf.co/solarkyle/jspace-lenses — the central community repo. Fits for gemma-4-E4B-it, gemma-4-12B-it ("most readable"), gemma-4-26B-A4B MoE, gemma-4-31B (deep-band read guidance), Qwen3.6-27B, gpt-oss-20b (harmony-format read note), Mistral-Small-24B. Fitted on 100 WikiText-103 prompts, bf16. Key claims to verify and cite: a bf16-fitted lens reads NF4 4-bit activations without refitting; **instructed lies: "the TRUE answer stays measurably elevated in the workspace while the output says otherwise"** (closest prior to this project); live hallucination detector + lie-detector UI at github.com/solarkyle/jspace. Contact in README invites collaboration. Email planned.
- [x] **zachperlman20/gemma-4-31B-it-jpp-lens** — https://hf.co/zachperlman20/gemma-4-31B-it-jpp-lens — J++ fit on the instruct checkpoint itself (pinned revision), layers 0-58, 3.4 GB fp16, documented layout (`lens["J"][layer]`), held-out evals beat the pooled lens (macro 0.633 vs 0.503). README states the base-model J++ fit (neuronpedia) failed a positive control on the instruct model: direct evidence for our exact-checkpoint rule. Verified from the repo README 2026-10-25.
- [x] **camilablank/workspace-lenses** — https://hf.co/camilablank/workspace-lenses — matched J-lens/R-lens pairs for 8 models including **Qwen/Qwen3.6-27B** (same forward pass, LRP-rule backward for R), plus a template lens. R-Lens post: lesswrong.com/posts/nv8oedrnLXKRzNEL9 (Blank & Bhatia); RelP rules from arXiv 2508.21258. Stacked-Jacobian dict layout (`lens["J"]`). Verified from the repo README 2026-10-25.
- [x] **bubskee/dolos-qwen25-workspace-lenses** — https://hf.co/bubskee/dolos-qwen25-workspace-lenses — Apart AI Agent Collusion Sprint team (BlueDot-funded compute): J/R/J++ lenses for Dolos-1.5B (undocumented community deception-finetune of Qwen2.5-1.5B-Instruct, Poseidon-Research, no model card) and its parent. Explicitly "not deception detectors"; no sandbagging or recovery results. Recall@10 on 1.5B: logit 35%, J 36-37%, R 42-45%, J++ 52%. Adjacent team to watch; also our read-only trained-in-vs-prompted exploratory arm. Verified 2026-10-25.
- [ ] Kameshr/jspace-lens-Llama8b — https://hf.co/Kameshr/jspace-lens-Llama8b — J-lens for meta-llama/Llama-3.1-8B-Instruct (1000 prompts, Hutchinson VJP). Cheap Llama-family addition once meta-llama access is granted.
- [ ] solarkyle traces — 500 TriviaQA q/a/correct per model (`traces/uncertainty_trivia_*.jsonl`), used by our bank filter; also fake-entity probes, NF4 quantization checks, covert-emotion readouts, frozen LightGBM hallucination classifiers (16,178 labelled responses, pre-registered, cross-dataset AUROC +0.065 over logprob baseline).
- [ ] solarkyle/jspace-hallucination-campaign (dataset) — 24,540 graded Gemma-4-12B responses with workspace features, pre-registered.
- [ ] **Kameshr/jspace-lens-Qwen7B** — https://hf.co/Kameshr/jspace-lens-Qwen7B — Qwen2.5-7B-Instruct fit, 1000 prompts, Hutchinson VJP estimator; own library github.com/kameshkanna/jspace (HookedModel / WorkspaceAnalyzer API, different from jlens). README carries the J_l maths used in our paper template.
- [ ] Festyve/jspace-lenses — deepseek-coder-1.3b-instruct fit, done overnight on an M4 laptop (~9 min/prompt, 40 prompts); independent implementation github.com/Festyve/jspace-viz validated vs the Neuronpedia Pythia lens (per-layer cosine 0.66–0.98). Useful as a template for fitting our own.
- [ ] jspace-emergence-paper/pythia-jlens-fits — lens fits across Pythia sizes; implies an emergence-across-scale paper exists. Find it.
- [ ] KikoCis/jspace-reproduction-qwen2.5-0.5b (dataset) — full reproduction of the workspace finding on a 0.5B model; proof the fitting route is cheap.
- [ ] kiruluta/J-Space-HF-Scaling-Benchmark — scaling/benchmark tooling. Inspect.
- [ ] Ziruibest/jspace-unlearning (dataset) — **lens-based audit of machine unlearning** (does "unlearned" knowledge persist), TOFU, Llama-3.2-1B, paper draft referenced. Structurally closest published design to ours.
- [ ] professorsynapse/eh-jspace-* (datasets) — epistemic humility / abstention work with gate AUROCs; multiple releases.
- [ ] Pixedar/jspace-strange-loop and jspace-ring (datasets) — self-model experiments (model reading descriptions of its own state), Qwen3.
- [ ] aravdhoot/jspace-distill-results — emergent-misalignment x distillation x LoRA on gemma-3-12b-it. Possibly relevant to our LoRA arm; inspect.
- [ ] jwest33/qwen3.5-9b-jspace-abliterated — "jspace-abliterated" uncensored model; implies someone used workspace directions for abliteration. Relevant to the causal-intervention story; inspect.
- [ ] guell00/J-Space-Deliberation, ceselder/modulation-lens-grid-* (Qwen3.6-27B LoRA adapters), Kimi-K2 and DeepSeek J-space entries (kimi-chan/Kimi-K2-Instruct-Jspace, ayh015/J-space, Juanpinha/j-space-lenses) — uninspected; triage.

## Sandbagging, deception, scheming literature

- [ ] van der Weij et al., *AI Sandbagging: Language Models can Strategically Underperform on Evaluations* (verify id ~2406.07358) — prompted sandbagging on MMLU/WMDP; source for instruction patterns.
- [ ] Greenblatt et al. (Redwood), *Stress-Testing Capability Elicitation With Password-Locked Models* (verify id ~2405.19550) — basis of our LoRA arm.
- [ ] MASK benchmark — arXiv 2503.03750, leaderboard scale.com/leaderboard/mask — belief pinned, then pressure to lie; borrow pressure framings.
- [ ] Apollo Research, in-context scheming evaluations — apolloresearch.ai/research/scheming-reasoning-evaluations — scenario-based propensity evals; source scenarios for the scheming follow-up.
- [ ] Deception probes: *Pressure-Testing Deception Probes in LLMs* (2605.27958); *One Probe Won't Catch Them All* (2602.01425); *LLMs Deceive Unintentionally* (2510.08211).
- [ ] Nguyen et al., *Probing and Steering Evaluation Awareness* (2507.01786) — my previous project's anchor; eval-awareness SAE steering partially uncovers sandbagging.
- [ ] Sharma et al., *Towards Understanding Sycophancy* (2310.13548); Arditi et al., refusal direction (verify ~2406.11717) — single-direction lineage for related work.

## Review-driven additions (2026-10-27, from the external design review)

- [x] **PIR / *A Lie Detector Test for Language Models: Reading Knowledge a Model Won't Reveal*** — arXiv 2609.21996 — title verified by search 2026-10-27. Per the review (of the 2026-09-23 version): causal steering tests, evaluation across model families, AUC-based concealment detection, trained and external concealment organisms, free-form experiments with reconstructed answer candidates (those details: verify against the paper). Closest method neighbour; the distinction is candidate-conditioned recognition versus direct pre-answer vocabulary readout, and our edge has to be practical (fewer forward passes, no candidate presentation, recovery when output sampling misses).
- [x] ***Eliciting Secret Knowledge from Language Models*** — arXiv 2510.01070 (also OpenReview 7SnghonEMY); MATS project with an Alignment Forum post — existence and venue verified by search 2026-10-27. Trained secret-keeping organisms; auditor-assisting methods including the logit lens and SAEs. Precedent for lens-based recovery of concealed information: do not claim prior lenses only classify.
- [x] ***Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized Deception*** — arXiv 2610.12445, formerly *Caught in the Act*; retitle verified by search 2026-10-27. Review corrections: the 0.63 concerns hidden-goal denials, not withheld-answer recovery; v4 (2026-10-20) reportedly adds an activation-oracle 0.80 and a probe rise 0.63 to 0.93 with a relevant preceding interaction (verify against v4). Not a ceiling our answer-recovery AUROC can be compared against.
- [ ] **SPAR Fall 2026, "The Signature of Scheming"** — sparai.org/projects/f26/rec8h4EE5tioWPoGA — page exists (verified); per the review it proposes sandbagging and concealment organisms via J-space, activation oracles, and transfer across training mechanisms (verify on page). Collaboration lead, and proof the space is occupied.
- [ ] ***J-space auditing might be unreliable*** — lesswrong.com/posts/69fachkoAs2ZHeSst/j-space-auditing-might-be-unreliable-1 (a sibling post exists at slug 7Qi3LR9ZXpvptLp6m) — existence verified by search 2026-10-27. Checkpoint and lens distribution-shift concerns; the prior for our parent-vs-refitted lens matrix.
- [ ] **Pranay Narula, public J-space readouts under instructed lying** (LinkedIn) — the review verified the public description only, no quantitative report; ask for the write-up before citing as evidence.
- Code address correction: the J++ implementation lives at **github.com/safety-research/jpp_lens** (the paper's own link); the older koayon/jpp_lens address does not resolve.
- Fresh triage surfaced while verifying the above (all unread, (verify)): *Auditing Games for Sandbagging* (arXiv 2512.07810, UK AISI); *Liars' Bench* (arXiv 2511.16035, NeurIPS 2026); *Same Question, Different Lies: C3 for Black-Box Sandbagging Detection* (PMLR v306); *Option-Order Randomisation Reveals a Distributional Position Attractor in Prompted Sandbagging* (arXiv 2604.26206).

## Incidents and motivation

- [ ] Redwood Research, *The OpenAI/Hugging Face Incident* — blog.redwoodresearch.org — July 2026 sandbox escape during a cyber eval; OpenAI and HF disclosures ~July 20-21, 2026. Motivation slide/intro citation.
- [ ] METR, *Recent Frontier Models Are Reward Hacking* (2025) — misbehaviour during evals, behavioural detection limits.

## Our artifacts (fill in as they exist)

- [ ] Repo: github.com/in-a-quantum-world/anthropic-jspace-scheming
- [ ] Pilot results: run branches, results/
- [ ] First pilot numbers (gemma-4-E4B-it, 2026-10-25): known 58/60, visible 58/58, comply 33/58, persist 31/33

## Triage queue (search these, nothing verified)

- [ ] arXiv/LessWrong sweep: "sandbagging" x lens/workspace/probing since July 2026
- [ ] Neuronpedia blog since April 2026 for J-lens features
- [ ] jacobian-lens GitHub issues/PRs for in-flight sandbagging work
