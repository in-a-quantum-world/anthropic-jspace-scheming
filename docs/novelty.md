# Novelty log

> **Correction (2026-10-27, after external review).** The review judged this log's central claim overstated and some neighbour descriptions incomplete. Specific corrections: PIR (arXiv 2609.21996) is substantially closer than described here, including causal steering, cross-family evaluation, concealment organisms and candidate-based recovery experiments; lens-based recovery of concealed information has precedent (*Eliciting Secret Knowledge*, arXiv 2510.01070), so "lenses only classify" is wrong; the 0.63 figure from *Caught in the Act* (now *Telling FIBS*, arXiv 2610.12445 v4) concerns hidden-goal denials and is not a ceiling our recovery AUROC competes with; SPAR Fall 2026 "The Signature of Scheming" announces directly overlapping work. The positioning has been replaced by the reliability-and-calibration framing in `docs/review-response.md`. The text below is kept as written, for the record.

The claim under test was that there is no published quantified account of sandbagged-knowledge recovery via J-space readouts. Anything that falsifies or narrows the claim is logged here with a date.

## Known adjacent work (as of 2026-10-25)

- **solarkyle/jspace-lenses README**: reports qualitatively that on *instructed lies* "the TRUE answer stays measurably elevated in the workspace while the output says otherwise", plus a lie-detector chat UI. Closest known prior. Not: sandbagging/eval framing, quantified recovery with baselines, causal ablation, trained-in (LoRA) concealment. Author invites collaboration (email in README) — contact planned.
- **Hallucination detection**: solarkyle campaign, 24,540 graded responses, pre-registered, frozen classifiers. Same design family, different behaviour.
- **Unlearning audit** (Ziruibest/jspace-unlearning): lens used to check whether "unlearned" knowledge persists. Structurally closest published design; different behaviour and no concealment incentive.
- **Epistemic humility / abstention** (professorsynapse), **self-model loops** (Pixedar): other claimed applications.
- Sandbagging literature (behavioural): van der Weij et al. prompted sandbagging; Redwood password-locked models (basis for the LoRA arm).

## To do, week 1

- [ ] Full arXiv / LessWrong / Alignment Forum sweep for "sandbagging" x internals/probing/lens
- [ ] Email solarkyle with the one-paragraph pitch (doubles as expert novelty check)
- [ ] Check Neuronpedia blog and jacobian-lens repo issues for in-flight work
