# Sandbagging in the J-space

When a language model deliberately underperforms on an evaluation, is the withheld knowledge still readable from its internal workspace, and can it be recovered?


## The question

Sandbagging (a model hiding what it knows during evaluation) undermines the validity of capability and safety evals. This project uses the Jacobian lens from Anthropic's J-space work to test whether sandbagged knowledge remains visible internally:

1. **Persistence (RQ1):** does the correct answer stay elevated in the workspace while the output is deliberately wrong?
2. **Recovery (RQ2):** what fraction of withheld answers can the lens read out, against norm-matched baselines?
3. **Causation (RQ3):** does ablating the concealment-related concepts restore honest answering?

Robustness arm: a password-locked LoRA where the sandbagging is trained into the weights rather than prompted, checking the monitor is not just reading the instruction text.

## Design

- Open-ended factual questions with single-token answers; the correct token never appears in the prompt, so anything found internally is model-generated rather than prompt echo.
- Honest vs sandbagged conditions on the same filtered question bank (hand-written items plus filtered TriviaQA items from the [solarkyle/jspace-lenses](https://huggingface.co/solarkyle/jspace-lenses) traces).
- Positive controls on every run: the lens must see answers the model gets right honestly, or the run is void.
- Confidence intervals on every proportion, analysis plan frozen before interventions run, a null is reported, not rescued.

## Models and tooling

| Model | Lens | Role |
| --- | --- | --- |
| google/gemma-4-E4B-it (4B) | solarkyle/jspace-lenses | pilot, free tier |
| google/gemma-4-12B-it | solarkyle/jspace-lenses | primary |
| Qwen/Qwen3.6-27B | solarkyle/jspace-lenses | cross-family check |

Lens library: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache 2.0). Lenses fitted in bf16 read 4-bit activations, so everything here runs on a single GPU.

## Gates

| Phase | Gate |
| --- | --- |
| Feasibility | lens loads; positive control: answers the model knows are visible in the workspace |
| Behaviour | sandbag compliance at a usable rate, else strengthen wording or escalate model |
| Readout | readable layer-by-layer traces, else restrict to top-k |
| Analysis | persistence or structural contrast, reported with lens-coverage caveats |
| Intervention | effects vs norm-matched baselines; a null is a result |
| Write-up | scope frozen |

## Repository layout

- `notebooks/` — runnable pipeline, starting with the hour-0 pilot
- `results/` — one folder per run (`run/YYYYMMDD-model` branches merge here); raw JSON plus a short notes file
- `docs/` — plan, gates, novelty log
- `src/` — batch scripts (phase 2 onward)

## Status

Phase 1 (feasibility pilot) in progress. Nothing in `results/` yet; the first entry will be the pilot's three gate numbers.
