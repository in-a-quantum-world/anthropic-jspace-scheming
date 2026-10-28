# Response to external design review (received 2026-10-27)

The review is accepted in substance. This file records what changes, what was re-analysed immediately, and what is deferred. The review's own scope note applies: it was a design-and-artifact review, not a code audit.

## Accepted immediately, with new numbers

1. **Within-question matched statistic** (review 3.2). Computed from the existing week-1 trajectories, J-lens, primary band, complied items:

| model | pooled AUROC | pooled, emitted retained | matched | matched, emitted retained | truth beats own lie |
|---|---|---|---|---|---|
| e4b | 0.762 | 0.709 | 0.732 | 0.685 | 2/27 |
| 12b | 0.398 | 0.370 | 0.407 | 0.377 | 2/35 |
| qwen | 0.718 | 0.682 | 0.729 | 0.694 | 3/29 |
| 31b | 0.400 | 0.371 | 0.363 | 0.343 | 2/33 |

Matched and pooled agree within 0.04 throughout, so pooling was not inflating the result.

2. **Emitted wrong answer retained** (review 3.3). The reviewer is right that the read occurs at the final prompt position, before generation, so the original exclusion rationale ("it is in the workspace because it was said") was wrong. Retained-emitted numbers above. The decisive observation is that the truth outranks the model's own planned lie on only 2-3 of ~30 complied items per model, so the readout is dominated by the planned output. The consequence for framing is that excluding the emitted answer is justified as the auditor's decoder, which conditions on the observed output and discounts it, rather than as a validity assumption. Both variants will be reported side by side from now on; neither replaces the other.

3. **Candidate budget** (review 3.1). "Top-20 at any band layer" has a worst-case budget of 20 x band layers (220-320 tokens across our models), not 20. Future recovery claims will state the realised shortlist size, and a fixed number of fractional-depth reads will be reported alongside the full-band statistic.

4. **H2 rephrased** (review 1). Honest-condition visibility is 0/55 in the first three quarter-bands, so the depth profile characterises the measurement procedure, not where the model represents the answer. H2 claims are about lens readability depth from here on.

5. **Summary claim downgraded** (review 1). Adopted: exploratory evidence that correct-answer tokens remain distinguishable from distractors during instructed underperformance, with visibility strongly dependent on read depth. "Recovering sandbagged knowledge" is the objective, not the result.

## Accepted, scheduled next

- **Fresh held-out bank** with audited distractors (answer type, plausibility, frequency, tokenisation, template variety) plus relation-changing controls; the current 55 items become development data. The prereg gains an amendment, not an edit (see prereg Amendment 1).
- **Distractor audit** of the current bank (the nitrogen item's non-gas elements being the worked example).
- **Baselines**: add final-output logits; treat the logit lens as a competitor, not a floor; report J++ under both a common ranking protocol and its native readout-filtered protocol; one budgeted black-box elicitation baseline.
- **Parent-vs-refitted-lens matrix on a password-locked organism** as the priority extension (review 4), with an honest-finetuning control, disjoint facts, and no evaluation answers in fitting. This is now the project's main experiment, under the review's framing of separating trained concealment from measurement failure.
- **Novelty positioning** revised per review 2: PIR (arXiv 2609.21996), Eliciting Secret Knowledge, Telling FIBS v4 (2610.12445), the SPAR Signature of Scheming project, and the Narula readout reports go into the reference pack; claims of priority are dropped in favour of the reliability-and-calibration positioning.
- **Repo hygiene**: status-accurate README, CLAIMS.md, exact prompt strings in versioned config, manifests, append-only deviation log, regeneration script for all tables.

## Noted corrections to our own earlier notes

- DeepSeek-V4-Flash: 284B total, 13B active, mixed FP4/FP8; block outputs are four parallel residual streams whose mean is the published lens's convention, so naive single-stream hooks will not reproduce it. Gate list per review 6; not the next milestone, and not the first custom lens fit.
- koayon/jpp_lens has moved to safety-research/jpp_lens.
- J++ expert-combination weights are fitted on labelled readout items; those labels must stay disjoint from test answers.
