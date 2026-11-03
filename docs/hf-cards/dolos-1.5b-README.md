---
license: apache-2.0
tags:
- jacobian-lens
- j-space
- interpretability
- qwen2.5
---

# J-lens for Dolos-1.5B (same-recipe refit)

A Jacobian (J-space) lens fitted by Rucha Agashe on 30 October 2026 to Dolos-1.5B, the community deception-finetune of Qwen/Qwen2.5-1.5B-Instruct distributed in [bubskee/dolos-qwen25-workspace-lenses](https://huggingface.co/bubskee/dolos-qwen25-workspace-lenses) (`models/dolos/`; fine-tune provenance undocumented upstream, a caveat this lens inherits).

Together with [RA-1-618/jspace-lens-qwen2.5-1.5b-instruct](https://huggingface.co/RA-1-618/jspace-lens-qwen2.5-1.5b-instruct) this forms a recipe-controlled parent/fine-tune lens pair: the official [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) estimator, the identical corpus (first 40 WikiText-103 records, max_seq_len 128) and identical settings, so the only difference between the two instruments is the checkpoint they were fitted on. Both were fitted on one RTX 5070 (8 GB). 27 source layers (0-26, target 27), d_model 1536.

Built for the project [anthropic-jspace-scheming](https://github.com/in-a-quantum-world/anthropic-jspace-scheming), where it rehearses the refit arm of the parent-fitted versus refitted lens matrix.

**Validation.** On the project's full 55-item development bank the lens reads 53/53 honestly stated answers, identical to bubskee's independently fitted Dolos J-lens on the same checkpoint. Sandbag-side numbers rest on 6 complied items (known 1.5B compliance) and are not quoted. Full details in the [lens card](https://github.com/in-a-quantum-world/anthropic-jspace-scheming/blob/main/docs/lens-cards/dolos-1.5b-jlens.md).

**Use.**

```python
from huggingface_hub import hf_hub_download
from jlens import JacobianLens
lens = JacobianLens.load(hf_hub_download("RA-1-618/jspace-lens-dolos-1.5b", "lens.pt"))
```

**Unsupported:** any other checkpoint, early-layer claims, and any sandbag-side quantitative claim at this compliance level.
