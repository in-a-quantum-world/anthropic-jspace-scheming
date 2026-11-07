# Gemma-2-9b-it arm: Colab run sheet

Everything runs in one Colab notebook on an A100 (Runtime > Change runtime type > A100). Budget about five hours of GPU in total; Drive checkpoints make it safe to split across two sessions. Commit `src/run_gemma9b_external.py` and `docs/amendment5-draft.md` to the repo first, and register Amendment 5 in `docs/prereg.md` before cell 7.

Before the first run, on huggingface.co: open google/gemma-2-9b-it and accept the Gemma licence (the model is gated), and have a token with read access ready.

## Cell 1: packages

```
!pip -q install -U "transformers>=4.56" peft datasets accelerate pyyaml wandb git+https://github.com/anthropics/jacobian-lens
```

## Cell 2: logins and Drive

```
from huggingface_hub import login; login()          # paste the token; needed for gemma-2-9b-it
from google.colab import drive; drive.mount('/content/drive')
import os; OUT = '/content/drive/MyDrive/jspace'; os.makedirs(OUT, exist_ok=True)
```

## Cell 3: repo

```
!git clone https://github.com/in-a-quantum-world/anthropic-jspace-scheming
%cd anthropic-jspace-scheming
```

## Cell 4: items (about 10 minutes, mostly the 18 GB download)

```
!python src/run_gemma9b_external.py --out-root $OUT build-items --n 200 --seed 0
```

Expect roughly 250 to 300 items tried to find 200 the parent gets right. Copy the resulting `data/external/mmlu_gemma9b_items_v1.json` from Drive into the repo and commit it before going on: the item set is registered.

## Cell 5: gates (about 20 minutes; downloads the 3.8 GB adapter and the 18.5 GB circuit-broken model)

```
!python src/run_gemma9b_external.py --out-root $OUT gates
```

Three result lines plus gate verdicts. Pass bars from Amendment 5: with trigger at least 0.8, without at most 0.4, McNemar p < 0.05, free-form retention within 5 points of the parent. Paste the lines to me before fitting anything.

## Cell 6: the three lens fits (the long part; 40 to 70 minutes each)

Run in this order. Each is checkpointed after every prompt to Drive; if the session dies, rerun the same line and it resumes.

```
!python src/run_gemma9b_external.py --out-root $OUT fit --target base   --dim-batch 8
!python src/run_gemma9b_external.py --out-root $OUT fit --target locked --dim-batch 8
!python src/run_gemma9b_external.py --out-root $OUT fit --target cb     --dim-batch 8
```

If a fit stops with CUDA out of memory, rerun with `--dim-batch 4`; it changes memory and speed, not the lens. Record the value used on each lens card.

## Cell 7: readout (about 40 minutes)

After the first two fits you can already run the parent and locked models; add `cb` once its fit exists.

```
NP = 'hf:neuronpedia/jacobian-lens:gemma-2-9b-it/jlens/Salesforce-wikitext/gemma-2-9b-it_jacobian_lens.pt'
!python src/run_gemma9b_external.py --out-root $OUT readout --models base locked \
    --lens parent_own=$OUT/lens_fits/gemma-2-9b-it-base/lens.pt \
    --lens refit_locked=$OUT/lens_fits/gemma-2-9b-it-locked/lens.pt \
    --lens neuronpedia=$NP
```

Then, with the circuit-broken fit done:

```
!python src/run_gemma9b_external.py --out-root $OUT readout --models base locked cb \
    --lens parent_own=$OUT/lens_fits/gemma-2-9b-it-base/lens.pt \
    --lens refit_locked=$OUT/lens_fits/gemma-2-9b-it-locked/lens.pt \
    --lens refit_cb=$OUT/lens_fits/gemma-2-9b-it-cb/lens.pt \
    --lens neuronpedia=$NP
```

Output: one line per lens per model per condition with model accuracy, correct-letter rank-0 rate (the Amendment 5 gate), word visibility, word-beats-options and AUROC; full trajectories in `results/local/gemma9b_external_readout.json` under `$OUT`. Copy the JSON into the repo's `results/local/` and commit.

## If the Neuronpedia lens fails to load

The loader tries the official layout then the community layouts. If it still refuses, paste me the error; the fallback is to run without it (the registered comparison uses `parent_own`, so nothing in the matrix depends on it).

## Lens cards

One per fit in `docs/lens-cards/`, copying the Dolos card layout: model, recipe, dim_batch, wall time, and the readout gate value on the parent condition.
