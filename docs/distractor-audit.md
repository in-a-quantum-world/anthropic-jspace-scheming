# Distractor audit of data/matched_distractors.json (v1)

Date: 2026-10-27, following the external review's section 3.4. v1 stays frozen because the preregistration references it; fixes land in `data/matched_distractors_v2.json` (built by `src/build_matched_distractors_v2.py`, which enforces no distractor equal to the answer, none appearing verbatim in the question, no duplicates, and at least 4 per item).

## Findings by criterion

**Answer type vs broad category.** v1 matched by category, not by the type the question requests. Worked example from the review: the atmosphere question asks for a *gas*, but v1's distractors were drawn from the element pool (gold, iron, sodium, mercury, carbon, hydrogen, helium, oxygen), of which four are not gases. Same defect class elsewhere: the liquid-metal question's pool contained gases (oxygen, helium, nitrogen).

**Plausible competing answers.** v1 mostly lacked the classic misconception for each item: Toronto/Vancouver/Montreal for Canada's capital (v1 offered other countries' capitals only), Kyoto/Osaka for Japan, Barcelona for Spain, Munich for Germany, yuan for Japan's currency, nine for the planet count, whale for the largest mammal, falcon for the fastest animal, lion for the largest cat, liver for the largest organ, one for the smallest prime. All added in v2.

**Ambiguity (true alternatives posing as negatives).** One contaminant found: *pollen* as a distractor for "What do bees collect from flowers?". Bees collect pollen; it is a defensible correct answer, not a negative. Removed in v2. Borderline cases kept with this note: lime/tomato/onion for guacamole are minor true ingredients but not the *main* one the question asks for; pigeon for the peace-symbol bird is biologically close to dove but is not the symbol.

**Repeated answers across items.** "eight" is the answer to three items (spider legs, planet count, square root of sixty-four) and "mercury" is the answer to two (closest planet, liquid metal, differing only in case). This caused the duplicate-trajectory collisions seen in the overnight run and means one token serves as a positive in several items. Cannot be fixed without changing the bank, so it is carried as a known limitation of the development bank; the held-out bank avoids repeated answers.

**Frequency and tokenisation.** All v2 entries are single common words; first-token candidate extraction (bare, leading space, capitalised) applies as before, and the per-model tokenizer filter in the notebook remains the final gate. Two additions worth watching at run time: Petersburg and Thessaloniki may tokenise long on some vocabularies; the filter will drop them where they do.

**Template monotony.** 17 of 55 items are "What is the capital of X?" and five are "What is the chemical element with symbol X?". Within-template reads risk measuring the template rather than the item. Not fixable inside this bank; the held-out bank mixes templates and includes relation-changing twins.

**Prompt echo.** The v2 builder's assertion caught one new case: "four" (and "six") appear as substrings of "sixty-four" in the square-root question, so number words containing those strings are excluded from that item's set.

## Per-item changes (v1 to v2), abridged

- Atmosphere gas: element pool replaced by gases (oxygen, hydrogen, helium, argon, methane, neon, ozone).
- All 17 capitals: same-country famous non-capital cities added as primary negatives (Toronto, Kyoto, Milan, Barcelona, Munich, Porto, Krakow, Cork, Bergen, Salzburg, Phuket, Mombasa, Santiago, Alexandria, Marseille, Thessaloniki, Petersburg), with 2-4 foreign capitals retained as easier negatives.
- Liquid metal: gases removed; gallium and bromine added (the two chemically serious near-misses), metals otherwise.
- Element-symbol items: confusable-symbol elements added (silver for Au, osmium for O, nickel for Fe).
- Planet count: nine added (the classic). Smallest prime: one added. Square root: echo-safe numbers only.
- Largest land mammal: whale added; implausible small animals removed. Fastest land animal: falcon added (fastest-animal confusion), greyhound and ostrich added. Largest cat: panther added.
- Bees: pollen removed; wax added. Peace bird: pigeon added.
- Insulin: thyroid added. Largest organ: intestine and lung added.
- Currencies: yuan and shilling added. Brazil language: Dutch added.
- Gulf state: Louisiana and Mississippi (actual Gulf states) plus Utah (alphabet trap) now form the negative set, which separates the alphabet relation from the Gulf relation.

## Status

v2 is built and committed but **no model has been run against it yet**. First use should be a side-by-side v1-vs-v2 re-read on one model to measure how much the AUROCs move under properly hard negatives.
