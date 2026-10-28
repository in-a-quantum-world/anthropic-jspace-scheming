# Run manifest: models, lenses, and code per run

Revisions were not captured at run time for the October runs, so fields marked (unlogged) can only be reconstructed as the current revision as of the run date. From the next run onward the notebook records `model.config._commit_hash` and lens file hashes into each results JSON.

| Run | Date | Models (id, revision) | Lenses (repo/file, revision) | Code |
|---|---|---|---|---|
| 20261025-0111-pilot | 2026-10-25 | google/gemma-4-E4B-it, google/gemma-4-12B-it, Qwen/Qwen3.6-27B (unlogged) | solarkyle/jspace-lenses: gemma-4-e4b-it, gemma-4-12b-it, qwen3.6-27b lens.pt (unlogged) | notebooks/sandbagging_jspace_pilot.ipynb at repo commit of that day |
| extended | 2026-10-25 | gemma-4-12B-it (unlogged) | solarkyle 12b lens | pilot notebook + added cell |
| 20261025-0118-sweep | 2026-10-25 | gemma-4-12B-it (unlogged) | solarkyle 12b lens | pilot notebook + sweep cell |
| 20261025-1052-week1 | 2026-10-25 | e4b, 12b, qwen, 31b (unlogged) | solarkyle lenses; koayon/jpp-lenses qwen3.6-27b; camilablank/workspace-lenses qwen3.6-27b j-lens and r-lens (31b zachperlman20 J++ configured but readouts disabled by the final-norm bug, fixed 671bf12-era) | notebooks/week1_followup.ipynb, pre-fix version (commit 4d396f2) |
| 20261025-1807-week1 (Dolos) | 2026-10-25 | Qwen/Qwen2.5-1.5B-Instruct; Dolos-1.5B from bubskee bundle (hashes in that repo's manifest.json) | bubskee/dolos-qwen25-workspace-lenses lenses/{parent,dolos}/{j,r,jpp} | same notebook, cell 7 |
| local fit, deepseek 1.3b | 2026-10-27 | deepseek-ai/deepseek-coder-1.3b-instruct (unlogged) | own fit, lens_fits/deepseek-coder-1.3b-instruct/lens.pt; reference Festyve/jspace-lenses | src/fit_lens_5070.py |

Quantisation per run: gemmas bf16 (e4b/12b), 31b bf16 on A100, qwen NF4; locals bf16. Prompt strings: configs/prompts.yaml.
