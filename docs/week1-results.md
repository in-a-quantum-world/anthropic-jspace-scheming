# Week-1 offline analysis: depth sweep and AUROC rebuild (gemma-4-12B-it)

Source: `results/20261025-0118-sweep/sweep_12b_main.json` (n_rows=55, dropped=0 duplicate/collision rows, 47 layers). Sandbag metrics on complied rows (n=55). The token the model actually emitted while sandbagging is excluded from distractor sets.

| band | AUROC | persist@5 | persist@20 | distractor@20 | honest vis@20 |
|---|---|---|---|---|---|
| pilot (0.25-1.0) | 0.889 | 7/55 | 27/55 | 0.071 | 55/55 |
| q1 (0-0.25) | 0.499 | 0/55 | 0/55 | 0.000 | 0/55 |
| q2 (0.25-0.5) | 0.499 | 0/55 | 0/55 | 0.000 | 0/55 |
| q3 (0.5-0.75) | 0.604 | 0/55 | 0/55 | 0.000 | 3/55 |
| q4 (0.75-1.0) | 0.893 | 7/55 | 27/55 | 0.071 | 55/55 |
| late (0.6-0.95) | 0.903 | 6/55 | 17/55 | 0.048 | 55/55 |
| no-tail (0.25-0.9) | 0.885 | 6/55 | 16/55 | 0.045 | 55/55 |

Figures: `paper/figures/depth_sweep_12b.*`, `paper/figures/traj_nitrogen_12b.*`.
Full sliding-window numbers: `results/analysis/depth_sweep_12b.json`.
