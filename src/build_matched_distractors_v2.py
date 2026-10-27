#!/usr/bin/env python3
"""Build v2 of the matched-distractor bank, after the 2026-10-27 external review.

v1 (data/matched_distractors.json) matched distractors by broad category; the
review's audit (docs/distractor-audit.md) found type mismatches (non-gas
elements for a gas question), missing classic misconceptions (Toronto for
Canada's capital, yuan for Japan's currency, nine for the planet count), and
one true-alternative contaminant (pollen for the bee question). v2 fixes those
with per-item curated sets. v1 is left untouched; the preregistration
references it.

Deterministic; no GPU. Output: data/matched_distractors_v2.json
Run from repo root: python3 src/build_matched_distractors_v2.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_matched_distractors import BANK  # noqa: E402

# Per-item curated distractors, keyed by exact question string.
# Design rules: match the requested answer TYPE (not just the broad category);
# include the classic misconception where one exists; exclude anything that is
# a defensible alternative answer; exclude anything appearing verbatim in the
# question; single-word, common-vocabulary entries only.
V2 = {
    "Which gas makes up most of Earth's atmosphere?":
        ["oxygen", "hydrogen", "helium", "argon", "methane", "neon", "ozone"],
    "What is the capital of France?":
        ["Marseille", "Lyon", "Nice", "Rome", "Madrid", "Berlin", "Brussels"],
    "What is the capital of Japan?":
        ["Kyoto", "Osaka", "Yokohama", "Beijing", "Seoul", "Bangkok"],
    "What is the capital of Italy?":
        ["Milan", "Naples", "Venice", "Florence", "Paris", "Madrid", "Athens"],
    "What is the capital of Spain?":
        ["Barcelona", "Seville", "Valencia", "Lisbon", "Paris", "Rome"],
    "What is the capital of Russia?":
        ["Petersburg", "Kazan", "Kiev", "Minsk", "Warsaw", "Berlin"],
    "What is the capital of Egypt?":
        ["Alexandria", "Giza", "Luxor", "Amman", "Tripoli", "Khartoum"],
    "What is the capital of Canada?":
        ["Toronto", "Vancouver", "Montreal", "Calgary", "Washington", "London"],
    "What is the capital of Germany?":
        ["Munich", "Hamburg", "Frankfurt", "Cologne", "Vienna", "Paris"],
    "What is the capital of Greece?":
        ["Thessaloniki", "Sparta", "Rome", "Istanbul", "Nicosia", "Sofia"],
    "What is the capital of Portugal?":
        ["Porto", "Braga", "Coimbra", "Madrid", "Seville", "Barcelona"],
    "What is the capital of Poland?":
        ["Krakow", "Gdansk", "Prague", "Berlin", "Vilnius", "Budapest"],
    "What is the capital of Ireland?":
        ["Cork", "Galway", "Belfast", "London", "Edinburgh", "Glasgow"],
    "What is the capital of Norway?":
        ["Bergen", "Trondheim", "Stockholm", "Copenhagen", "Helsinki"],
    "What is the capital of Austria?":
        ["Salzburg", "Graz", "Innsbruck", "Zurich", "Munich", "Prague", "Budapest"],
    "What is the capital of Thailand?":
        ["Phuket", "Pattaya", "Hanoi", "Manila", "Jakarta", "Singapore"],
    "What is the capital of Kenya?":
        ["Mombasa", "Kisumu", "Kampala", "Lagos", "Accra", "Dodoma"],
    "What is the capital of Cuba?":
        ["Santiago", "Varadero", "Kingston", "Miami", "Caracas", "Managua"],
    "What is the largest planet in the solar system?":
        ["Saturn", "Neptune", "Uranus", "Mars", "Venus", "Mercury"],
    "Which planet is closest to the sun?":
        ["Venus", "Mars", "Earth", "Jupiter", "Pluto"],
    "Which planet is known as the red planet?":
        ["Venus", "Jupiter", "Mercury", "Saturn", "Neptune"],
    "What is the chemical element with symbol Au?":
        ["silver", "copper", "iron", "platinum", "tin", "lead", "zinc"],
    "What is the chemical element with symbol Fe?":
        ["copper", "zinc", "lead", "tin", "gold", "silver", "nickel"],
    "What is the chemical element with symbol O?":
        ["hydrogen", "nitrogen", "carbon", "helium", "osmium", "neon"],
    "What is the chemical element with symbol He?":
        ["hydrogen", "neon", "argon", "oxygen", "nitrogen"],
    "What is the chemical element with symbol Na?":
        ["potassium", "calcium", "magnesium", "lithium", "nitrogen", "neon"],
    "What metal is liquid at room temperature?":
        ["gallium", "lead", "tin", "silver", "sodium", "bromine"],
    "How many legs does a spider have?":
        ["six", "four", "ten", "twelve", "two"],
    "How many sides does a hexagon have?":
        ["five", "seven", "eight", "four", "three"],
    "How many continents are there on Earth?":
        ["five", "six", "eight", "nine", "four"],
    "How many planets are in the solar system?":
        ["nine", "seven", "ten", "six", "twelve"],
    "What is the longest river in the world?":
        ["Amazon", "Yangtze", "Mississippi", "Congo", "Danube"],
    "What is the tallest mountain on Earth?":
        ["Kilimanjaro", "Denali", "Fuji", "Matterhorn", "Olympus", "Annapurna"],
    "What is the largest ocean on Earth?":
        ["Atlantic", "Indian", "Arctic", "Southern"],
    "What is the smallest prime number?":
        ["one", "three", "zero", "five", "seven"],
    "What is the square root of sixty-four?":
        ["seven", "nine", "twelve", "five", "ten"],
    "What animal is the largest living land mammal?":
        ["rhino", "hippo", "giraffe", "bear", "bison", "whale"],
    "What is the fastest land animal?":
        ["leopard", "lion", "greyhound", "antelope", "ostrich", "horse", "falcon"],
    "What is the largest species of big cat?":
        ["lion", "leopard", "jaguar", "cheetah", "panther"],
    "What bird is the universal symbol of peace?":
        ["pigeon", "eagle", "swan", "owl", "crane"],
    "What is the main ingredient of guacamole?":
        ["lime", "tomato", "onion", "pepper", "cucumber", "bean"],
    "What fruit is traditionally associated with Isaac Newton?":
        ["pear", "plum", "cherry", "fig", "orange"],
    "Which organ pumps blood around the human body?":
        ["liver", "brain", "kidney", "spleen", "stomach"],
    "Which organ in the human body produces insulin?":
        ["liver", "kidney", "stomach", "spleen", "thyroid"],
    "What is the human body's largest organ?":
        ["liver", "heart", "brain", "intestine", "lung"],
    "What is frozen water called?":
        ["snow", "frost", "hail", "sleet", "steam"],
    "What do bees collect from flowers?":
        ["honey", "dew", "sap", "resin", "wax"],
    "In which country are the pyramids of Giza?":
        ["Mexico", "Sudan", "Peru", "Jordan", "Morocco"],
    "In which country is the Eiffel Tower?":
        ["Italy", "Belgium", "Spain", "Germany", "England"],
    "In which country is Mount Fuji?":
        ["China", "Korea", "Nepal", "Vietnam", "Taiwan"],
    "Which language is spoken in Brazil?":
        ["Spanish", "French", "Italian", "English", "Dutch"],
    "What currency is used in Japan?":
        ["yuan", "won", "euro", "dollar", "rupee"],
    "What currency is used in the United Kingdom?":
        ["euro", "dollar", "franc", "shilling", "krone"],
    "In which country is the Sky Train Rail bridge?":
        ["America", "Australia", "England", "Japan", "India"],
    "Which state on the Gulf of Mexico is nearest the end of the alphabet?":
        ["Utah", "Florida", "Alabama", "Louisiana", "Mississippi", "Georgia"],
}


def main():
    items = []
    for q, a, cat in BANK:
        dis = V2[q]
        ql = q.lower()
        for d in dis:
            assert d.lower() != a.lower(), (q, d, "distractor equals answer")
            assert d.lower() not in ql, (q, d, "distractor appears in question")
        assert len(dis) >= 4, (q, dis)
        assert len(dis) == len({d.lower() for d in dis}), (q, "duplicate distractors")
        items.append({"q": q, "a": a, "category": cat, "distractors": dis})
    assert len(items) == len(V2) == 55
    os.makedirs("data", exist_ok=True)
    with open("data/matched_distractors_v2.json", "w") as f:
        json.dump({"version": 2, "max_distractors": 8, "items": items,
                   "note": "Curated per-item hard negatives after the 2026-10-27 review; "
                           "see docs/distractor-audit.md. v1 is frozen for the prereg."}, f, indent=1)
    print(f"wrote data/matched_distractors_v2.json ({len(items)} items)")


if __name__ == "__main__":
    main()
