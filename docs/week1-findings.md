# Week-1 confirmatory run: findings (run 20261025-1052-week1)

Four models, matched distractors, full trajectories, primary band 0.75-1.0. All positive controls passed (honest visibility 100% on every model and every readout, including both J++ fits, camilablank J/R and the logit lens on qwen).

## Registered results (prereg H1/H4, matched-distractor AUROC in primary band)

| model | known | comply | persist@20 | AUROC (J-lens) |
|---|---|---|---|---|
| gemma-4-E4B-it | 53/55 | 31/53 | 29/31 (0.94) | 0.762 |
| gemma-4-12B-it | 55/55 | 55/55 | 27/55 (0.49) | 0.398 |
| Qwen3.6-27B | 50/55 | 48/50 | 43/48 (0.90) | 0.718 |
| gemma-4-31B-it | 53/55 | 53/53 | 11/53 (0.21) | 0.400 |

H1 (AUROC >= 0.75 on e4b, 12b, qwen): **fails as registered** (e4b passes, qwen 0.718 narrowly misses, 12b clearly fails). H4 (12b matched AUROC >= 0.70): **fails**. Reported as registered; no threshold is revised post hoc.

## Why: matched distractors are partially positives

Same-category neighbours are elevated in the workspace even under the honest condition (category priming by the question), and markedly more under sandbagging (lie search):

| model | neighbours@20, honest | neighbours@20, sandbag | delta |
|---|---|---|---|
| e4b | 0.554 | 0.776 | +0.22 |
| 12b | 0.286 | 0.617 | +0.33 |
| qwen | 0.538 | 0.707 | +0.17 |
| 31b | 0.122 | 0.227 | +0.10 |

Interpretation: when instructed to give a wrong one-word answer, the model's candidate lies are exactly the true answer's category neighbours. A rank-only monitor cannot cleanly separate "withheld truth" from "considered lies" within the category. The random-distractor baseline (overnight runs: 12b AUROC 0.89) and the matched baseline bracket the monitor's real operating range and are both reported.

Two claims survive and sharpen:

1. **Recovery**: the withheld truth stays readable, persist@20 0.90 on qwen and 0.94 on e4b. The monitor recovers the answer; it struggles to certify which elevated same-category token is the truth.
2. **Lie-search signature**: the honest-to-sandbag jump in neighbour elevation is itself a detectable signature of sandbagging as a condition, independent of recovering the answer. New, not in any neighbouring work.

## Lens robustness (qwen, five readouts, same items)

cb_r 0.732, cb_j 0.730, lens (solarkyle J) 0.718, jpp 0.714, logit 0.664. All transported lenses agree within 0.02; the logit floor trails. The conclusion is not a one-lens artifact.

## Open items

- 31b: truth barely readable in any band (sliding-band max AUROC 0.57, persist@20 max 11/53) with a perfect positive control. Bigger-gemma concealment or weaker lens fit; the zachperlman20 J++ control did not run in this session (hidden-state readouts were disabled on the gemma models; error text to be pulled from the Colab cell-5 output) and is the disambiguator.
- Earliness (depth of first elevation) alone does not separate truth from lie candidates (AUROC 0.54-0.68); richer trajectory-shape features are untested.
- Dolos exploratory arm: pending (cell 7).

## Dolos exploratory arm (run 20261025-1807-week1, read-only)

Positive controls perfect (50/50 honest visibility on logit, J, R, J++ for both Dolos-1.5B and its Qwen2.5-1.5B-Instruct parent). But compliance was 6/50 on both models, and several "complied" items were scoring artifacts (digit "6" for "six"; "Icy"/"Icicle" for "ice"), so the printed AUROCs (0.81-0.90) rest on too few items to quote. Two usable observations:

1. Instructed-sandbagging competence scales: compliance ~10% at 1.5B, 58% at 4B (31/53), ~100% at 12B and above. Deliberate underperformance on instruction is itself a capability; concealment detection is a capable-model problem.
2. Dolos's deception tuning gave no extra instructed-sandbagging ability over its parent (6/50 vs 6/50), so it does not serve the trained-in-vs-prompted contrast; the controlled arm remains necessary.

Compliance scoring now maps number words to digit forms (notebook fix in this commit); re-scored compliance for 1.5B models is lower than the printed 6/50.

## Rerun 20261028-0934 (fixed readouts, digit-aware compliance)

The final-norm fix works: gemma models now have logit baselines. e4b: logit AUROC 0.781 vs J-lens 0.762 (persist 31/31 vs 29/31), so at 4B the logit lens is a genuine competitor, as the external review predicted; on qwen the transported lenses still lead (0.705-0.722 vs logit 0.656). Qwen's five readouts reproduce the 2026-10-25 numbers within 0.01 and the gemma J-lens numbers reproduce exactly. Caveat: the 12b logit readout fails the 90% positive-control gate (46/55), so its sandbag numbers are flagged. 31b was not included in this run (default model list); its J++ disambiguation remains the top open item. (The previous commit carried this run's result files; this one carries the analysis text.)
