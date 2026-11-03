---
license: apache-2.0
tags:
- jacobian-lens
- j-space
- interpretability
- qwen2.5
---

# J-lens for Qwen/Qwen2.5-1.5B-Instruct

A Jacobian (J-space) lens fitted by Rucha Agashe on 28 October 2026 with the official [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) estimator: exact per-layer Jacobian rows, first 40 WikiText-103 records of at least 600 characters (`jlens.examples.load_wikitext_prompts(40)`), max_seq_len 128, fitted in bf16 on one RTX 5070 (8 GB). 27 source layers (0-26, target 27), d_model 1536, fp16 on disk.

This is J_parent for the parent-fitted versus refitted lens matrix in the project [anthropic-jspace-scheming](https://github.com/in-a-quantum-world/anthropic-jspace-scheming); the same-recipe refit on a fine-tune of this checkpoint is at [RA-1-618/jspace-lens-dolos-1.5b](https://huggingface.co/RA-1-618/jspace-lens-dolos-1.5b).

**Validation.** Functional positive control 10/10 at fit time; convergence max_d_mean 0.23 at prompt 8 falling to 0.03 at prompt 40. On the project's full 55-item development bank the lens reads 51/51 honestly stated answers, identical to two independently fitted community lenses for the same checkpoint, with sandbag-side AUROC 0.786 against their 0.789 and 0.770. Full details in the [lens card](https://github.com/in-a-quantum-world/anthropic-jspace-scheming/blob/main/docs/lens-cards/qwen2.5-1.5b-instruct-jlens.md).

**Use.**

```python
from huggingface_hub import hf_hub_download
from jlens import JacobianLens
lens = JacobianLens.load(hf_hub_download("RA-1-618/jspace-lens-qwen2.5-1.5b-instruct", "lens.pt"))
```

**Unsupported:** any other checkpoint, quantised activations (untested for this fit), and early-layer claims (band outside 0.75-1.0 fractional depth unvalidated).
