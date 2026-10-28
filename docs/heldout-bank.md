# Held-out bank (data/heldout_bank_v1.json)

Created 2026-10-27 per the external review's section 3.5. 48 new single-word-answer factual questions, none sharing a question or an answer with the 55-item development bank; 16 items are relation-changing control twins (`control_twin_of`): similar entities and vocabulary to their partner, but a different relation determines the answer, so a readout that merely follows the question's associations scores the twin wrong.

**Isolation rule.** No model has been run on this bank as of its creation. It stays out of LoRA training data, lens-fitting corpora, decoder and layer selection, and threshold calibration. First contact with any model is the confirmatory run itself, with metrics and thresholds fixed beforehand (prereg Amendment 1 and its successors). Any earlier contact must be logged in docs/analysis-log.md and disqualifies the touched items from confirmatory use.

Built by `src/build_heldout_bank.py`, which enforces: no question/answer overlap with the development bank, answers and distractors absent from their question text (substring check), no duplicate answers within the bank, at least 4 distractors per item, and at least 15 control twins. The substring check rejected two drafts during construction (vitamin "D" against any word containing d; "euro" against "European"), which confirms that the check operates as intended.
