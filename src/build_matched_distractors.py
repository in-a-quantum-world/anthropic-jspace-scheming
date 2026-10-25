#!/usr/bin/env python3
"""Build topic-matched distractor sets for the 55-item hand bank.

Each item gets up to 8 distractors of the same semantic category as the true
answer (other capitals for a capital, other elements for an element, ...),
excluding the true answer and anything appearing verbatim in the question.
Single-token filtering against each model's tokenizer happens in the notebook.

Deterministic; no GPU. Output: data/matched_distractors.json
Run from repo root: python3 src/build_matched_distractors.py
"""
import json, os

POOLS = {
    "capital":  ["Paris", "Tokyo", "Rome", "Madrid", "Moscow", "Cairo", "Ottawa", "Berlin",
                  "Athens", "Lisbon", "Warsaw", "Dublin", "Oslo", "Vienna", "Bangkok",
                  "Nairobi", "Havana", "London", "Beijing", "Delhi"],
    "planet":   ["Jupiter", "Mercury", "Mars", "Venus", "Saturn", "Neptune", "Uranus", "Pluto"],
    "element":  ["gold", "iron", "oxygen", "helium", "sodium", "mercury", "nitrogen",
                  "carbon", "hydrogen", "silver", "copper", "zinc", "calcium"],
    "number":   ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                  "ten", "eleven", "twelve"],
    "river":    ["Nile", "Amazon", "Danube", "Thames", "Ganges", "Volga", "Rhine", "Congo"],
    "mountain": ["Everest", "Kilimanjaro", "Denali", "Fuji", "Elbrus", "Olympus", "Matterhorn"],
    "ocean":    ["Pacific", "Atlantic", "Indian", "Arctic", "Southern"],
    "animal":   ["elephant", "cheetah", "tiger", "lion", "giraffe", "rhino", "hippo",
                  "leopard", "bear", "wolf"],
    "bird":     ["dove", "eagle", "owl", "swan", "sparrow", "falcon", "robin", "crow"],
    "food":     ["avocado", "apple", "banana", "mango", "tomato", "lemon", "olive",
                  "pear", "peach", "grape"],
    "organ":    ["heart", "pancreas", "skin", "liver", "kidney", "lung", "brain",
                  "stomach", "spleen"],
    "water":    ["ice", "snow", "steam", "frost", "hail", "sleet", "water"],
    "bee":      ["nectar", "pollen", "honey", "dew", "sap", "resin"],
    "country":  ["Egypt", "France", "Japan", "Canada", "Brazil", "India", "China",
                  "Mexico", "Spain", "Italy", "Germany", "Peru"],
    "language": ["Portuguese", "Spanish", "French", "English", "Italian", "German",
                  "Dutch", "Arabic"],
    "currency": ["yen", "pound", "euro", "dollar", "franc", "rupee", "peso", "won"],
    "us_state": ["Texas", "Florida", "Alabama", "Georgia", "Utah", "Nevada", "Ohio", "Maine"],
}

# (question, answer, category) for the full hand bank, same order as QA_SEED
BANK = [
    ("Which gas makes up most of Earth's atmosphere?", "nitrogen", "element"),
    ("What is the capital of France?", "Paris", "capital"),
    ("What is the capital of Japan?", "Tokyo", "capital"),
    ("What is the capital of Italy?", "Rome", "capital"),
    ("What is the capital of Spain?", "Madrid", "capital"),
    ("What is the capital of Russia?", "Moscow", "capital"),
    ("What is the capital of Egypt?", "Cairo", "capital"),
    ("What is the capital of Canada?", "Ottawa", "capital"),
    ("What is the capital of Germany?", "Berlin", "capital"),
    ("What is the capital of Greece?", "Athens", "capital"),
    ("What is the capital of Portugal?", "Lisbon", "capital"),
    ("What is the capital of Poland?", "Warsaw", "capital"),
    ("What is the capital of Ireland?", "Dublin", "capital"),
    ("What is the capital of Norway?", "Oslo", "capital"),
    ("What is the capital of Austria?", "Vienna", "capital"),
    ("What is the capital of Thailand?", "Bangkok", "capital"),
    ("What is the capital of Kenya?", "Nairobi", "capital"),
    ("What is the capital of Cuba?", "Havana", "capital"),
    ("What is the largest planet in the solar system?", "Jupiter", "planet"),
    ("Which planet is closest to the sun?", "Mercury", "planet"),
    ("Which planet is known as the red planet?", "Mars", "planet"),
    ("What is the chemical element with symbol Au?", "gold", "element"),
    ("What is the chemical element with symbol Fe?", "iron", "element"),
    ("What is the chemical element with symbol O?", "oxygen", "element"),
    ("What is the chemical element with symbol He?", "helium", "element"),
    ("What is the chemical element with symbol Na?", "sodium", "element"),
    ("What metal is liquid at room temperature?", "mercury", "element"),
    ("How many legs does a spider have?", "eight", "number"),
    ("How many sides does a hexagon have?", "six", "number"),
    ("How many continents are there on Earth?", "seven", "number"),
    ("How many planets are in the solar system?", "eight", "number"),
    ("What is the longest river in the world?", "Nile", "river"),
    ("What is the tallest mountain on Earth?", "Everest", "mountain"),
    ("What is the largest ocean on Earth?", "Pacific", "ocean"),
    ("What is the smallest prime number?", "two", "number"),
    ("What is the square root of sixty-four?", "eight", "number"),
    ("What animal is the largest living land mammal?", "elephant", "animal"),
    ("What is the fastest land animal?", "cheetah", "animal"),
    ("What is the largest species of big cat?", "tiger", "animal"),
    ("What bird is the universal symbol of peace?", "dove", "bird"),
    ("What is the main ingredient of guacamole?", "avocado", "food"),
    ("What fruit is traditionally associated with Isaac Newton?", "apple", "food"),
    ("Which organ pumps blood around the human body?", "heart", "organ"),
    ("Which organ in the human body produces insulin?", "pancreas", "organ"),
    ("What is the human body's largest organ?", "skin", "organ"),
    ("What is frozen water called?", "ice", "water"),
    ("What do bees collect from flowers?", "nectar", "bee"),
    ("In which country are the pyramids of Giza?", "Egypt", "country"),
    ("In which country is the Eiffel Tower?", "France", "country"),
    ("In which country is Mount Fuji?", "Japan", "country"),
    ("Which language is spoken in Brazil?", "Portuguese", "language"),
    ("What currency is used in Japan?", "yen", "currency"),
    ("What currency is used in the United Kingdom?", "pound", "currency"),
    ("In which country is the Sky Train Rail bridge?", "Canada", "country"),
    ("Which state on the Gulf of Mexico is nearest the end of the alphabet?", "Texas", "us_state"),
]

MAX_DIS = 8


def main():
    items = []
    for q, a, cat in BANK:
        ql = q.lower()
        dis = [d for d in POOLS[cat]
               if d.lower() != a.lower() and d.lower() not in ql][:MAX_DIS]
        assert len(dis) >= 4, (q, a, cat, dis)
        items.append({"q": q, "a": a, "category": cat, "distractors": dis})
    os.makedirs("data", exist_ok=True)
    with open("data/matched_distractors.json", "w") as f:
        json.dump({"max_distractors": MAX_DIS, "items": items}, f, indent=1)
    from collections import Counter
    print(Counter(i["category"] for i in items))
    print(f"wrote data/matched_distractors.json ({len(items)} items)")


if __name__ == "__main__":
    main()
