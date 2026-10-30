# Some definitions!

• Sandbagging: a model deliberately underperforming on an evaluation, hiding what it knows or can do. In my runs so far it is instructed, meaning the prompt tells the model to give a wrong answer; the next phase trains it into the weights instead.

• Interpretability: methods that read a model's internal computation, its hidden states, rather than only its outputs. This project is partly a test of whether one interpretability instrument can be trusted for auditing.

• Workspace and the Jacobian lens (J-lens): the lens maps each layer's hidden state into vocabulary space through a linear map estimated from the model's own Jacobians, fitted once on generic text. The workspace is this layer-by-layer view of what the model is considering; J-space is Anthropic's name for it.

• Logit lens: the zero-fitting baseline readout: apply the model's own final unembedding directly to intermediate layers. It costs nothing, and at 4B it currently matches the J-lens, which is why I treat it as a competitor rather than a floor.

• R-lens: a lens variant built with different backward attribution rules. It is independently fitted by another group, and I use it to check that conclusions do not depend on a single instrument.

• J++: a newer variant that routes each readout through cluster-specific expert Jacobians instead of one averaged map, generally a sharper instrument. It is the cross-check running on the 31B anomaly.

• Matched-distractor question bank: 55 open-ended factual questions with single-word answers, hand-written plus filtered TriviaQA items. The correct token never appears in the prompt, so anything found internally is model-generated rather than an echo. Each item carries a category and topic-matched distractors drawn from the same category as the truth; if the answer is Jupiter, the distractors are other planets. Matching makes the test deliberately hard, because random distractors would be easy to beat.

• Positive control gate: on every run the lens must see at least 90% of the answers the model gives correctly in the honest condition, or the run is void. It establishes that the instrument works before any negative reading is interpreted.

• Compliance: the fraction of items where the model actually obeys the sandbag instruction and emits a wrong answer. Only complied items enter the sandbag-side metrics.

• Persistence: the weakest useful outcome: the withheld correct answer still appears in the lens's top 20 tokens in the primary read band during sandbagging.

• Recovery: the stronger outcome: a rule fixed in advance actually picks the answer out, from a candidate set or from the open vocabulary, without being told it. Persistence without recovery means the signal is present but cannot yet be certified.

• AUROC: the probability that a randomly chosen true answer outranks a randomly chosen distractor in the readout. 1.0 is perfect separation, 0.5 is chance; my registered thresholds were written in terms of it.

• Held-out: data quarantined from all development. The 48-item held-out bank has never been run through any model and is excluded from lens fitting, decoder selection and threshold calibration; it is used once, for the confirmatory numbers.

• Frozen: fixed before the results existed and never edited afterwards. The preregistration text and the first bank are frozen; changes exist only as numbered amendments or new versioned files.

• Parent-fitted vs refitted lens: the parent lens is fitted to the original model; the refitted lens applies the identical recipe to the fine-tuned model. Crossing the two over both models' activations, with an honest-finetuning control, is the 2 by 4 matrix at the heart of the project.


