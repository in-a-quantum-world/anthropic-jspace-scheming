# Can Workspace Lenses Recover Withheld Answers? Separating Trained Concealment from Measurement Failure

Rucha Agashe. BlueDot Technical AI Safety Project Sprint write-up, 28 October 2026.
Repository: github.com/in-a-quantum-world/anthropic-jspace-scheming. Every number below is traceable to CLAIMS.md and regenerates from saved results with `python3 src/regenerate_tables.py`.

## 1. The question

Safety cases lean heavily on evaluations, and evaluations are only informative if models cannot hide what they know. A model that deliberately underperforms on an evaluation (sandbagging) undermines them from the inside. My project evaluates the reliability and practical utility of workspace-lens answer recovery under prompted and, next, trained capability concealment, separating decoding success from lens calibration failures. Put more directly, when a model withholds an answer, can a Jacobian-lens readout of its residual stream recover that answer without knowing it in advance, and when recovery fails, is that because the model concealed the information or because the lens stopped measuring the model reliably?

The design is deliberately simple. I use 55 open-ended factual questions with single-token answers, where the correct token never appears in the prompt, so anything found internally is model-generated rather than prompt echo. Each model answers under an honest instruction and under a sandbag instruction ("give a deliberately WRONG one-word answer"). For every item I record the rank of the true answer and of topic-matched distractors in the lens readout at every layer, in both conditions. Every run includes a positive control, and if a lens cannot see answers the model is openly giving, its readings are not used.

Following an external design review, I keep five outcomes separate, because a project can succeed at one and fail at the others: persistence (the token appears in the readout), candidate recovery (a frozen rule picks the answer from a candidate set), open-vocabulary recovery (no gold answer supplied), concealment detection (withholding versus error), and behavioural elicitation (an intervention makes the model answer). The results below are persistence and candidate-separation results. Recovery remains the objective of the project rather than one of its current claims.

## 2. A misleading pilot interpretation I caught

My first pilot initially looked convincing. On gemma-4-12B-it, with distractors drawn randomly from other questions, the withheld answer separated from the distractors at AUROC 0.889 in the registered read band. It would have been easy to present that as an 89% recovery result.

It is not a recovery result. When I preregistered the design and replaced random distractors with topic-matched ones (other capitals for a capital question, other elements for an element question), the 12B result collapsed to AUROC 0.398. The reason turned out to be one of the main findings, which is that matched distractors are partially positives. When a model is told to answer wrongly, its candidate lies are exactly the true answer's category neighbours. Neighbour elevation in the readout jumps from the honest to the sandbag condition on every model (0.286 to 0.617 on 12B; 0.554 to 0.776 on 4B; 0.538 to 0.707 on 27B), which is the model searching for a plausible lie, visible in its internals.

The clearest statement of the problem is that the model's own planned wrong answer outranks the withheld truth on all but 2 or 3 of roughly 30 complied items per model. A naive decoder that reads the top of the lens recovers the planned wrong answer rather than the truth. Operationally, recovery therefore means discounting the model's observed output, which an auditor can legitimately do because they have the output in front of them. I report both analysis variants (emitted lie excluded, the auditor's decoder; and emitted lie retained, the raw separation problem, which costs 0.03 to 0.05 AUROC), and a within-question matched statistic agrees with the pooled one within 0.04 on all four models.

My preregistered thresholds H1 and H4 failed as written, and they are reported as failures.

## 3. The controls that distinguished explanations

**Positive-control gating.** Honest-condition visibility was 100% on every model and J-lens pair used (53/53, 55/55, 50/50, 53/53), so no sandbag claim rests on a lens that cannot read openly stated answers. One readout failed the gate (the 12B logit lens, 46/55 in the fixed rerun) and its sandbag numbers are flagged and not used.

**Registered results (exploratory development bank, J-lens, primary band 0.75 to 1.0):**

| model | known | comply | persist@20 | AUROC (matched) |
|---|---|---|---|---|
| gemma-4-E4B-it (4B) | 53/55 | 31/53 | 29/31 (0.94) | 0.762 |
| gemma-4-12B-it | 55/55 | 55/55 | 27/55 (0.49) | 0.398 |
| Qwen3.6-27B | 50/55 | 48/50 | 43/48 (0.90) | 0.718 |
| gemma-4-31B-it | 53/55 | 53/53 | 11/53 (0.21) | 0.400 |

**Not a one-lens artifact.** On the 27B I repeated every reading through five independently built readouts, including two community J-lens fits from different authors, an R-lens (same forward pass, different backward rules), and a J++ lens:

| readout (Qwen3.6-27B) | honest visibility | AUROC |
|---|---|---|
| camilablank R-lens | 50/50 | 0.732 |
| camilablank J-lens | 50/50 | 0.730 |
| solarkyle J-lens | 50/50 | 0.718 |
| koayon J++ | 50/50 | 0.714 |
| logit lens | 50/50 | 0.664 |

All transported lenses agree within 0.02, and a full rerun three days later reproduced these numbers within 0.01.

**The depth profile belongs to the measurement procedure.** A sliding-band sweep on 12B put all recoverable signal in the last quarter of layers (quarter bands one to three at AUROC 0.499, 0.499 and 0.604 with zero persistence). But honest-condition controls also fail outside the late band (0/55, 0/55, 3/55 visibility), so this characterises where the lens can read, not where the model represents the answer. I initially wrote the stronger claim and corrected it.

**Simple baselines are competitive.** At 4B the plain logit lens matches the J-lens (0.781 against 0.762); at 27B the transported lenses lead it by 0.05 to 0.07. The transported lens is not automatically better, and I now carry final-output logits and the logit lens as standing baselines.

## 4. Frozen and held-out

The preregistration is timestamped in the public repository. Its original text is frozen; the changes forced by the external review were added as a numbered amendment, so the record shows what changed and why. The amendment reclassifies the 55-item bank and everything computed on it as development evidence.

Confirmatory claims will use a fresh held-out bank: 48 new items with audited, type-matched distractors and 16 relation-changing control twins. No model has been run on it, and it is excluded from LoRA training, lens fitting, decoder selection and threshold calibration. The distractor audit also produced a corrected v2 of the development distractors (the original nitrogen item had non-gas elements as distractors; v2 uses gases).

Reproducibility machinery, added after the review: one command regenerates every reported table from saved results; exact prompt strings and read bands live in versioned configuration; metric definitions live in one shared module with unit tests covering AUROC tie handling and token matching; claims live in an append-only register with a limitation attached to each; the analysis log and run manifest record what ran, when, and with which artifacts.

## 5. Reusable artifacts

The repository is public and contains full per-layer rank trajectories for four models (4B to 31B), both conditions, every item, so others can test their own decoders without a GPU. The question bank, matched distractors (frozen v1 and audited v2) and the held-out bank are released with it.

I also fitted my own lens. The pipeline was validated by refitting the published deepseek-coder-1.3b lens with the identical recipe, which reproduced it to per-layer cosine 1.000, confirming the plumbing rather than independent convergence. I then fitted a J-lens for Qwen2.5-1.5B-Instruct on my own 8 GB consumer GPU, documented it in a lens card, and published it at huggingface.co/RA-1-618/jspace-lens-qwen2.5-1.5b-instruct. On the full bank it matches two independently fitted community lenses for the same checkpoint: honest visibility 51/51 through all readouts, sandbag AUROC 0.786 against their 0.789 and 0.770. This lens serves as J_parent, the parent-model instrument for the next experiment.

## 6. What remains unknown, and what comes next

**The 31B anomaly.** On gemma-4-31B-it the withheld answer is barely readable (persistence 11/53, AUROC 0.400) despite a perfect positive control, and no read band rescues it (maximum 0.57 anywhere). Either larger gemmas suppress withheld answers more thoroughly, which would be an important result, or the community lens fit is weak exactly where it matters. An independently fitted J++ lens exists for this checkpoint, and obtaining that second opinion is the most important open item; the run is queued.

**The main next experiment.** The next contribution is reliable elicitation under trained concealment. A password-locked fine-tune of the parent model (suppressed without a trigger phrase, capable with it, validated on held-out capability, with an honest-finetuning control) is read with both a parent-fitted and a checkpoint-refitted lens across all four activation conditions. If the parent lens fails where a refitted lens recovers answers, apparent information loss was measurement mismatch. If both read the answers despite wrong outputs, that is robust evidence of readable withheld information. If both fail while the trigger still unlocks the capability, this readout family fails under trained concealment. The operational endpoint is candidate recovery with a frozen decoder, evaluated on the held-out bank.

**A secondary observation.** Competence at instructed sandbagging scales with model size: compliance is roughly 10% at 1.5B, 58% at 4B, and near 100% from 12B up. Small models mostly cannot execute deliberate underperformance even when instructed to, which makes concealment detection primarily a problem involving capable models.

DeepSeek-V4-Flash, with its existing lens fits, remains a later replication with its analysis fixed in advance rather than the next step, and nothing in this report should be read as a deployable sandbagging detector.

## Limitations

All quantitative results above are exploratory, on a development bank of 55 single-token factual items in one prompt format, with instructed sandbagging standing in for the real behaviour; trained or self-generated concealment may differ. A top-20-at-any-band-layer criterion has a candidate budget of 20 times the band layers, not 20. The emitted lie was tracked only where it coincided with a tracked distractor; systematic tracking is planned. Absence from a readout is not absence from the model, so persistence nulls, including the 31B result, are lens-limited until cross-checked. The 1.5B sandbag-side numbers rest on 7 complied items. Community lens quality is mitigated by positive controls and multi-lens agreement, not eliminated.
