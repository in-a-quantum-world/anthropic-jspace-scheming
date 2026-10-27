#!/usr/bin/env python3
"""Build the fresh held-out evaluation bank (data/heldout_bank_v1.json).

Created 2026-10-27 after the external review: the 55-item development bank has
informed design decisions, so confirmatory claims need questions no model in
this project has seen. 48 new single-word-answer items, none sharing a question
or an answer with the development bank; 16 are relation-changing control twins
(similar entities and vocabulary, different relation determining the answer),
marked with control_twin_of.

ISOLATION RULE: no model output, lens fit, LoRA training run, decoder choice or
threshold may touch this bank until the confirmatory run itself. See
docs/heldout-bank.md.

Deterministic; no GPU. Run from repo root: python3 src/build_heldout_bank.py
"""
import json
import os

# id, question, answer, category, distractors, twin-of (None for standalone)
ITEMS = [
    # Twin pair 1: planets; ring fame vs morning-star name
    ("h01", "Which planet is famous for its bright rings?", "Saturn", "planet",
     ["Jupiter", "Neptune", "Uranus", "Venus", "Pluto"], None),
    ("h02", "Which planet is called the morning star?", "Venus", "planet",
     ["Saturn", "Jupiter", "Mars", "Sirius"], "h01"),
    # Twin pair 2: metals; conductivity vs melting point
    ("h03", "Which metal is the best conductor of electricity?", "silver", "metal",
     ["copper", "gold", "platinum", "titanium", "nickel"], None),
    ("h04", "Which metal has the highest melting point?", "tungsten", "metal",
     ["titanium", "platinum", "steel", "chromium", "cobalt"], "h03"),
    # Twin pair 3: months; god-origin vs leap day
    ("h05", "Which month is named after the Roman god of doorways and beginnings?",
     "January", "month", ["March", "June", "May", "April", "December"], None),
    ("h06", "Which month gains an extra day in a leap year?", "February", "month",
     ["January", "March", "December", "April", "June"], "h05"),
    # Twin pair 4: rivers; flows-through vs longest-in
    ("h07", "Which river flows through Rome?", "Tiber", "river",
     ["Danube", "Rhine", "Seine", "Thames", "Arno"], None),
    ("h08", "What is the longest river in Europe?", "Volga", "river",
     ["Danube", "Rhine", "Loire", "Elbe", "Thames"], "h07"),
    # Twin pair 5: mountains; eruption event vs highest-in
    ("h09", "Which volcano erupted and buried Pompeii?", "Vesuvius", "mountain",
     ["Etna", "Olympus", "Fuji", "Denali", "Matterhorn"], None),
    ("h10", "What is the highest mountain in Africa?", "Kilimanjaro", "mountain",
     ["Kenya", "Atlas", "Denali", "Elbrus", "Etna"], "h09"),
    # Twin pair 6: oceans; location vs size
    ("h11", "Which ocean lies between Africa and Australia?", "Indian", "ocean",
     ["Atlantic", "Arctic", "Southern", "Mediterranean"], None),
    ("h12", "Which ocean is the smallest?", "Arctic", "ocean",
     ["Atlantic", "Indian", "Southern", "Baltic"], "h11"),
    # Twin pair 7: organs; urine vs bile
    ("h13", "Which organ removes waste from the blood to make urine?", "kidney", "organ",
     ["liver", "bladder", "stomach", "spleen", "intestine"], None),
    ("h14", "Which organ produces bile?", "liver", "organ",
     ["kidney", "stomach", "pancreas", "spleen", "heart"], "h13"),
    # Twin pair 8: elements; fusion fuel vs diamond composition
    ("h15", "Which element powers the sun through nuclear fusion?", "hydrogen", "element",
     ["helium", "carbon", "uranium", "oxygen", "nitrogen"], None),
    ("h16", "Which element is diamond made of?", "carbon", "element",
     ["silicon", "quartz", "nitrogen", "boron", "hydrogen"], "h15"),
    # Twin pair 9: countries; animal home vs capital-of
    ("h17", "Which country is home to the kangaroo?", "Australia", "country",
     ["Austria", "Brazil", "India", "China"], None),
    ("h18", "Which country has Ankara as its capital?", "Turkey", "country",
     ["Greece", "Syria", "Iran", "Iraq"], "h17"),
    # Twin pair 10: deserts; continent coverage vs largest-in-Asia
    ("h19", "Which desert covers much of northern Africa?", "Sahara", "desert",
     ["Gobi", "Kalahari", "Mojave", "Namib", "Atacama"], None),
    ("h20", "Which desert in Asia is the largest?", "Gobi", "desert",
     ["Sahara", "Kalahari", "Mojave", "Thar", "Atacama"], "h19"),
    # Twin pair 11: colors; gemstone color vs color mixing
    ("h21", "What color are emeralds?", "green", "color",
     ["blue", "red", "yellow", "purple", "white"], None),
    ("h22", "What color do you get by mixing red and blue?", "purple", "color",
     ["green", "orange", "brown", "pink"], "h21"),
    # Twin pair 12: instruments; key count vs largest brass
    ("h23", "Which instrument has eighty-eight keys?", "piano", "instrument",
     ["organ", "guitar", "violin", "harp", "accordion"], None),
    ("h24", "What is the largest instrument in the brass family?", "tuba", "instrument",
     ["trumpet", "trombone", "horn", "cornet", "flute"], "h23"),
    # Twin pair 13: sports; venue vs signature move
    ("h25", "Which sport is played at Wimbledon?", "tennis", "sport",
     ["cricket", "golf", "badminton", "squash", "football"], None),
    ("h26", "In which sport would you perform a slam dunk?", "basketball", "sport",
     ["volleyball", "netball", "handball", "baseball"], "h25"),
    # Twin pair 14: birds; largest vs famous mimic
    ("h27", "What is the largest living bird?", "ostrich", "bird",
     ["emu", "eagle", "condor", "albatross", "penguin"], None),
    ("h28", "Which bird is famous for mimicking human speech?", "parrot", "bird",
     ["crow", "raven", "pigeon", "canary", "sparrow"], "h27"),
    # Twin pair 15: English cities; landmark vs band origin
    ("h29", "In which city is Big Ben?", "London", "city",
     ["Manchester", "Birmingham", "Edinburgh", "Liverpool", "Dublin"], None),
    ("h30", "Which English city is famous as the Beatles' home?", "Liverpool", "city",
     ["Manchester", "London", "Leeds", "Bristol", "Birmingham"], "h29"),
    # Twin pair 16: insects; makes product vs destroys wood
    ("h31", "Which insect makes honey?", "bee", "insect",
     ["wasp", "ant", "beetle", "moth", "fly"], None),
    ("h32", "Which insect is known for destroying wooden buildings?", "termite", "insect",
     ["ant", "beetle", "cockroach", "moth", "wasp"], "h31"),
    # Standalone items
    ("h33", "What force pulls objects toward the Earth?", "gravity", "physics",
     ["magnetism", "friction", "inertia", "momentum"], None),
    ("h34", "What is the hardest natural substance?", "diamond", "substance",
     ["quartz", "granite", "steel", "titanium", "ruby"], None),
    ("h35", "Which nut is the main ingredient of marzipan?", "almond", "food",
     ["walnut", "peanut", "cashew", "hazelnut", "pistachio"], None),
    ("h36", "Which yellow fruit do monkeys famously eat?", "banana", "fruit",
     ["mango", "papaya", "pineapple", "orange", "melon"], None),
    ("h37", "Which flower is the traditional symbol of love?", "rose", "flower",
     ["tulip", "lily", "daisy", "orchid", "violet"], None),
    ("h38", "Which bird is the fastest in a dive?", "falcon", "bird",
     ["eagle", "hawk", "swift", "ostrich", "owl"], None),
    ("h39", "Which sea creature has eight arms?", "octopus", "sea-animal",
     ["squid", "jellyfish", "starfish", "crab", "lobster"], None),
    ("h40", "What is the largest mammal in the ocean?", "whale", "sea-animal",
     ["shark", "dolphin", "seal", "walrus"], None),
    ("h41", "What is Earth's only natural satellite called?", "moon", "astronomy",
     ["Titan", "Europa", "Phobos", "Io"], None),
    ("h42", "Which language has the most native speakers in the world?", "Mandarin", "language",
     ["English", "Spanish", "Hindi", "Arabic", "French"], None),
    ("h43", "Which currency replaced the French franc and the German mark?", "euro", "currency",
     ["dollar", "krone", "zloty", "rouble"], None),
    ("h44", "Which gemstone is deep red?", "ruby", "gemstone",
     ["sapphire", "emerald", "topaz", "amethyst", "opal"], None),
    ("h45", "What is the tallest animal in the world?", "giraffe", "animal",
     ["elephant", "ostrich", "camel", "horse", "moose"], None),
    ("h46", "Which fabric gave its name to the ancient trade route from China to Europe?",
     "silk", "fabric", ["cotton", "wool", "linen", "nylon", "velvet"], None),
    ("h47", "Which drink is made from fermented grapes?", "wine", "drink",
     ["beer", "cider", "whisky", "vodka", "rum"], None),
    ("h48", "In which US state is Hollywood?", "California", "us_state",
     ["Nevada", "Florida", "Arizona", "Oregon"], None),
]

DEV_BANK_PATH = "data/matched_distractors.json"


def main():
    dev = json.load(open(DEV_BANK_PATH))["items"]
    dev_qs = {it["q"].lower() for it in dev}
    dev_as = {it["a"].lower() for it in dev}

    ids = [i[0] for i in ITEMS]
    assert len(ids) == len(set(ids))
    out = []
    n_twins = 0
    for iid, q, a, cat, dis, twin in ITEMS:
        ql = q.lower()
        assert q.lower() not in dev_qs, (iid, "question overlaps development bank")
        assert a.lower() not in dev_as, (iid, a, "answer overlaps development bank")
        assert a.lower() not in ql, (iid, "answer appears in question")
        for d in dis:
            assert d.lower() != a.lower(), (iid, d)
            assert d.lower() not in ql, (iid, d, "distractor appears in question")
        assert len(dis) >= 4, (iid, dis)
        assert len(dis) == len({d.lower() for d in dis}), (iid, "duplicate distractors")
        if twin is not None:
            assert twin in ids, (iid, twin)
            n_twins += 1
        item = {"id": iid, "q": q, "a": a, "category": cat, "distractors": dis}
        if twin is not None:
            item["control_twin_of"] = twin
        out.append(item)
    answers = [it["a"].lower() for it in out]
    assert len(answers) == len(set(answers)), "repeated answers within held-out bank"
    assert n_twins >= 15, n_twins
    os.makedirs("data", exist_ok=True)
    with open("data/heldout_bank_v1.json", "w") as f:
        json.dump({"version": 1, "max_distractors": 8, "items": out,
                   "note": "HELD OUT: no model has been run on this bank. Keep it out of "
                           "LoRA training, lens fitting, decoder selection and threshold "
                           "calibration until the confirmatory run. docs/heldout-bank.md."},
                  f, indent=1)
    print(f"wrote data/heldout_bank_v1.json ({len(out)} items, {n_twins} control twins)")


if __name__ == "__main__":
    main()
