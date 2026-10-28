# Novelty log

> **Correction (2026-10-27, after external review).** The review judged this log's central claim overstated and some neighbour descriptions incomplete. Specific corrections: PIR (arXiv 2609.21996) is substantially closer than described here, including causal steering, cross-family evaluation, concealment organisms and candidate-based recovery experiments; lens-based recovery of concealed information has precedent (*Eliciting Secret Knowledge*, arXiv 2510.01070), so "lenses only classify" is wrong; the 0.63 figure from *Caught in the Act* (now *Telling FIBS*, arXiv 2610.12445 v4) concerns hidden-goal denials and is not a ceiling our recovery AUROC competes with; SPAR Fall 2026 "The Signature of Scheming" announces directly overlapping work. The positioning has been replaced by the reliability-and-calibration framing in `docs/review-response.md`. The text below was restyled for prose on 2026-10-28; its content and claims are unchanged, and the original wording remains in the git history.

The claim under test was that there is no published quantified account of sandbagged-knowledge recovery via J-space readouts. Anything that falsifies or narrows the claim is logged here with a date.

## Known adjacent work (as of 2026-10-25)

- **solarkyle/jspace-lenses README**: reports qualitatively that on *instructed lies* "the TRUE answer stays measurably elevated in the workspace while the output says otherwise", together with a lie-detector chat UI. This is the closest known prior work. It does not include a sandbagging or evaluation framing, quantified recovery with baselines, causal ablation, or trained-in (LoRA) concealment. The author invites collaboration (email in the README), and contact is planned.
- **Hallucination detection**: the solarkyle campaign, 24,540 graded responses, pre-registered, with frozen classifiers. The same design family applied to a different behaviour.
- **Unlearning audit** (Ziruibest/jspace-unlearning): a lens used to check whether "unlearned" knowledge persists. This is the structurally closest published design, on a different behaviour and without a concealment incentive.
- **Epistemic humility / abstention** (professorsynapse) and **self-model loops** (Pixedar): other claimed applications of the same readout.
- Sandbagging literature (behavioural): van der Weij et al. on prompted sandbagging; the Redwood password-locked models, which are the basis for the LoRA arm.

## To do, week 1

- [ ] Full arXiv / LessWrong / Alignment Forum sweep for "sandbagging" x internals/probing/lens
- [ ] Email solarkyle with the one-paragraph pitch (doubles as an expert novelty check)
- [ ] Check the Neuronpedia blog and jacobian-lens repo issues for work in progress
