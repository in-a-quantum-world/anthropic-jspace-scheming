# Claims register

One row per claim: the claim, the artifact that supports it, and its main limitation. Claims not in this file are not made. Append-only; corrections add rows rather than editing old ones.

| # | Claim | Supporting artifact | Limitation |
|---|---|---|---|
| 1 | On gemma-4-E4B-it, gemma-4-12B-it, Qwen3.6-27B and gemma-4-31B-it, honest-condition answers are visible in the J-lens top-20 (late band) for 100% of capability-filtered items; the same holds for the logit, J++, and camilablank J/R readouts where run | `results/20261025-1052-week1/`, `docs/week1-findings.md` | Positive control only; late band; single prompt format |
| 2 | During complied instructed sandbagging, the withheld correct answer appears in the J-lens top-20 (band 0.75-1.0) on 94% (e4b), 90% (qwen), 49% (12b), 21% (31b) of items | same | Development bank; top-20-at-any-band-layer has a candidate budget of 20 x band layers, not 20 |
| 3 | Pooled AUROC vs topic-matched distractors: 0.76 (e4b), 0.72 (qwen), 0.40 (12b), 0.40 (31b); registered thresholds H1/H4 fail as written | same; `docs/prereg.md` | Matched distractors are partially positives (candidate lies); development bank |
| 4 | Within-question matched AUROC agrees with pooled within 0.04 on all four models | `docs/review-response.md` | Same data, descriptive |
| 5 | The model's own emitted wrong answer outranks the withheld truth on all but 2-3 of ~30 complied items per model; raw top-of-readout decoding recovers the lie, not the truth | `docs/review-response.md` | Emitted answer tracked only where it coincided with a distractor; systematic tracking planned |
| 6 | Same-category neighbour elevation jumps 10-33 points from honest to sandbag conditions | `docs/week1-findings.md` | Signature of lie-search under instruction; untested for trained concealment |
| 7 | Compliance with the sandbag instruction scales with size: ~10% at 1.5B, 58% at 4B, ~100% at 12B+ | `docs/week1-findings.md` | Instructed setting only; 1.5B numbers partly scoring-limited |
| 8 | A same-recipe refit of the deepseek-coder-1.3b lens reproduces the published community fit (per-layer cosine ~1.000) | `src/fit_lens_5070.py` output, 2026-10-27 | Validates the pipeline, not independent convergence: recipe and data were deliberately identical |
