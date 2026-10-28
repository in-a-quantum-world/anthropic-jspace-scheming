# Lens card: J-lens for Qwen/Qwen2.5-1.5B-Instruct (self-fitted)

- **Fitted by:** Rucha Agashe, 2026-10-28, on one RTX 5070 (8 GB), Windows.
- **Model:** Qwen/Qwen2.5-1.5B-Instruct, bf16 (revision not pinned at fit time; future fits pin it).
- **Estimator:** official anthropics/jacobian-lens `jlens.fit` (exact per-layer Jacobian rows, dim-batched VJPs; no stochastic estimator).
- **Corpus:** first 40 WikiText-103 (raw, train) records of at least 600 characters via `jlens.examples.load_wikitext_prompts(40)`; max_seq_len 128; first 16 positions skipped; n_valid 111 positions per prompt.
- **Config notes:** prompts 1-7 fitted at dim_batch 16, prompts 8-40 at dim_batch 8 after an out-of-memory fix. dim_batch only chunks the VJP computation and does not change the computed Jacobians, so the mix is harmless. Early prompts were computed during sysmem-spill episodes (slow, not wrong).
- **Convergence:** running-mean relative change (`max_d_mean`) 0.23 at prompt 8 falling to 0.03 at prompt 40. max||J||/sqrt(d) stable in 1.27-2.28.
- **Output:** 27 source layers (0-26, target layer 27), d_model 1536, fp16 on disk, ~130 MB (`lens_fits/qwen2.5-1.5b-instruct/lens.pt`, not in git; to be published to Hugging Face).
- **Validation:** functional positive control 10/10 (honest answers visible in top-20, band 0.75-1.0); logit-lens baseline also 10/10 on the same items. No same-recipe reference exists for this model, so no cosine comparison; the bubskee community lenses for this model are a planned independent cross-check on the full bank.
- **Intended use:** parent-model rows of the parent-vs-refitted lens matrix (P1), and local sandbagging evaluation. **Unsupported:** any other checkpoint, quantised activations (untested for this fit), early-layer claims (band outside 0.75-1.0 unvalidated).
