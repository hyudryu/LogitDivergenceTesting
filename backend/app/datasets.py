"""Built-in, license-free prompt sets, generated deterministically.

Divergence prompts reuse a small pool of original paragraphs, expanded with
seeded sentence shuffling to hit approximate context-length targets. NIAH
(needle-in-a-haystack) cases embed a synthetic needle at controlled depths.
"""
from __future__ import annotations

import random
from typing import Dict, List

SUITE_INFO = {
    "divergence": "Teacher-forced logprob scan + greedy generation divergence (pairwise)",
    "mcq": "Multiple-choice accuracy via next-token letter logprobs",
    "niah": "Synthetic needle-in-a-haystack long-context recall",
    "genqa": "Short-form QA with exact/numeric match scoring",
}

# --- Original prose used to build long prompts ------------------------------

PASSAGES = [
    (
        "Keeping a sourdough starter healthy is mostly a matter of rhythm. "
        "A starter is a stable culture of wild yeast and lactic acid bacteria that "
        "feeds on flour and water, and it rewards consistency. Feed it equal weights "
        "of flour and water once a day at room temperature, or once a week if it "
        "lives in the refrigerator. A mature starter smells pleasantly tangy, doubles "
        "within four to eight hours of feeding, and shows a web of bubbles across the "
        "surface. If a gray liquid collects on top, the starter is hungry, not spoiled; "
        "pour it off and feed again. Always reserve a portion of the old culture when "
        "you feed, because that acidified environment is what keeps unwanted microbes "
        "from taking hold."
    ),
    (
        "A bicycle drivetrain wears faster than most riders expect. The chain "
        "slowly elongates as the bushings and rollers inside it wear, and a stretched "
        "chain grinds away the teeth of the cassette and chainrings, turning a twenty "
        "dollar replacement into a two hundred dollar one. Checking chain wear with a "
        "gauge every few hundred miles is cheaper than replacing a cassette. Clean the "
        "chain with a degreaser, let it dry, then apply lubricant one drop per roller "
        "while backpedaling, and wipe off the excess. In wet weather, relubricate every "
        "hundred miles; in dry dust, wipe the chain down after every ride. A quiet, "
        "smooth drivetrain is usually a well measured one."
    ),
    (
        "The intertidal zone is one of the most demanding habitats on the coast. "
        "Organisms that live there are submerged at high tide and exposed to sun, "
        "wind, and predators at low tide, sometimes twice a day. Barnacles cement "
        "themselves to rock and close their plates to hold moisture. Sea stars use "
        "hundreds of tube feet to cling tightly enough that waves cannot pull them "
        "loose. Hermit crabs carry borrowed shells to protect soft abdomens. Zonation "
        "is visible as bands of color on the rocks: barnacles higher up, mussels and "
        "goose barnacles in the middle, and kelp and anemones only in pools that "
        "never fully drain. Disturbing that arrangement can take years to repair."
    ),
]

FILLER_SENTENCES = [
    "The meeting notes were filed in the shared drive before noon.",
    "A light rain fell steadily through the entire afternoon.",
    "The warehouse crew restacked the pallets near the loading dock.",
    "She revised the schedule twice before sending it out.",
    "The old printer in the hallway jammed on glossy paper.",
    "Coffee from the second pot tasted noticeably better.",
    "The ferry arrived ten minutes ahead of schedule.",
    "A pair of jays argued loudly over the bird feeder.",
    "The spreadsheet needed one more column for totals.",
    "Street sweepers passed through the district early on Tuesday.",
    "The museum extended its hours for the summer season.",
    "Moss covered the shaded northern edge of the stone wall.",
    "The recipe called for far more butter than expected.",
    "He labeled every box before the movers arrived.",
    "The trail forked just past the second bridge.",
    "Traffic on the overpass thinned out after seven.",
    "The library added study rooms to the third floor.",
    "A persistent draft rattled the window in the office.",
    "The gardener mulched the beds before the first frost.",
    "Their shuttle bus shortcut saved almost twenty minutes.",
]

MCQ_QUESTIONS = [
    {"id": "mcq01", "question": "What is the capital of Australia?",
     "options": {"A": "Sydney", "B": "Melbourne", "C": "Canberra", "D": "Perth"}, "answer": "C"},
    {"id": "mcq02", "question": "Which planet is known as the Red Planet?",
     "options": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Mercury"}, "answer": "B"},
    {"id": "mcq03", "question": "What is 17 x 24?",
     "options": {"A": "388", "B": "398", "C": "408", "D": "418"}, "answer": "C"},
    {"id": "mcq04", "question": "Which gas do plants absorb from the air for photosynthesis?",
     "options": {"A": "Oxygen", "B": "Nitrogen", "C": "Carbon dioxide", "D": "Hydrogen"}, "answer": "C"},
    {"id": "mcq05", "question": "What is the chemical symbol for gold?",
     "options": {"A": "Go", "B": "Gd", "C": "Au", "D": "Ag"}, "answer": "C"},
    {"id": "mcq06", "question": "How many continents are there on Earth?",
     "options": {"A": "Five", "B": "Six", "C": "Seven", "D": "Eight"}, "answer": "C"},
    {"id": "mcq07", "question": "Which ocean is the largest by area?",
     "options": {"A": "Atlantic", "B": "Indian", "C": "Arctic", "D": "Pacific"}, "answer": "D"},
    {"id": "mcq08", "question": "What is the square root of 169?",
     "options": {"A": "11", "B": "12", "C": "13", "D": "14"}, "answer": "C"},
    {"id": "mcq09", "question": "Who wrote the play Romeo and Juliet?",
     "options": {"A": "Charles Dickens", "B": "William Shakespeare", "C": "Jane Austen", "D": "Mark Twain"}, "answer": "B"},
    {"id": "mcq10", "question": "Which element has atomic number 1?",
     "options": {"A": "Helium", "B": "Hydrogen", "C": "Lithium", "D": "Oxygen"}, "answer": "B"},
    {"id": "mcq11", "question": "What is the largest animal on Earth?",
     "options": {"A": "African elephant", "B": "Blue whale", "C": "Giraffe", "D": "Orca"}, "answer": "B"},
    {"id": "mcq12", "question": "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
     "options": {"A": "60", "B": "70", "C": "80", "D": "90"}, "answer": "C"},
    {"id": "mcq13", "question": "Which language has the most native speakers worldwide?",
     "options": {"A": "English", "B": "Spanish", "C": "Mandarin Chinese", "D": "Hindi"}, "answer": "C"},
    {"id": "mcq14", "question": "What is the freezing point of water in degrees Fahrenheit?",
     "options": {"A": "0", "B": "32", "C": "100", "D": "212"}, "answer": "B"},
    {"id": "mcq15", "question": "Which mountain is the tallest above sea level?",
     "options": {"A": "K2", "B": "Kangchenjunga", "C": "Mount Everest", "D": "Denali"}, "answer": "C"},
    {"id": "mcq16", "question": "What is 15 percent of 200?",
     "options": {"A": "15", "B": "20", "C": "30", "D": "45"}, "answer": "C"},
    {"id": "mcq17", "question": "In which year did World War II end?",
     "options": {"A": "1943", "B": "1944", "C": "1945", "D": "1946"}, "answer": "C"},
    {"id": "mcq18", "question": "What is the primary function of red blood cells?",
     "options": {"A": "Fighting infection", "B": "Carrying oxygen", "C": "Clotting blood", "D": "Digesting fat"}, "answer": "B"},
    {"id": "mcq19", "question": "Which country is home to the kangaroo?",
     "options": {"A": "New Zealand", "B": "South Africa", "C": "Australia", "D": "Argentina"}, "answer": "C"},
    {"id": "mcq20", "question": "What is the next prime number after 7?",
     "options": {"A": "9", "B": "10", "C": "11", "D": "13"}, "answer": "C"},
    {"id": "mcq21", "question": "Which instrument measures atmospheric pressure?",
     "options": {"A": "Hygrometer", "B": "Barometer", "C": "Anemometer", "D": "Thermometer"}, "answer": "B"},
    {"id": "mcq22", "question": "How many sides does a hexagon have?",
     "options": {"A": "Five", "B": "Six", "C": "Seven", "D": "Eight"}, "answer": "B"},
    {"id": "mcq23", "question": "Which organelle is called the powerhouse of the cell?",
     "options": {"A": "Ribosome", "B": "Nucleus", "C": "Mitochondria", "D": "Golgi apparatus"}, "answer": "C"},
    {"id": "mcq24", "question": "Which data structure operates last-in, first-out?",
     "options": {"A": "Queue", "B": "Stack", "C": "Linked list", "D": "Binary heap"}, "answer": "B"},
]

GENQA_QUESTIONS = [
    {"id": "qa01", "question": "What is 12 + 30?", "answer": "42"},
    {"id": "qa02", "question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"id": "qa03", "question": "What is 8 x 7?", "answer": "56"},
    {"id": "qa04", "question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"id": "qa05", "question": "You have 3 boxes with 15 pencils each. How many pencils in total?", "answer": "45"},
    {"id": "qa06", "question": "What is the chemical formula for water?", "answer": "H2O"},
    {"id": "qa07", "question": "What is 100 divided by 4?", "answer": "25"},
    {"id": "qa08", "question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"id": "qa09", "question": "What is the speed of light in vacuum, rounded to the nearest hundred thousand kilometers per second?", "answer": "300,000"},
    {"id": "qa10", "question": "What is 45 + 55?", "answer": "100"},
    {"id": "qa11", "question": "On which continent is the Sahara Desert?", "answer": "Africa"},
    {"id": "qa12", "question": "What is the smallest prime number?", "answer": "2"},
    {"id": "qa13", "question": "What is 9 squared?", "answer": "81"},
    {"id": "qa14", "question": "How many minutes are in 2 hours?", "answer": "120"},
    {"id": "qa15", "question": "Which gas makes up about 78 percent of Earth's atmosphere?", "answer": "Nitrogen"},
    {"id": "qa16", "question": "What is 144 divided by 12?", "answer": "12"},
    {"id": "qa17", "question": "What is the boiling point of water in degrees Celsius?", "answer": "100"},
    {"id": "qa18", "question": "How many days are in a leap year?", "answer": "366"},
]

NIAH_DEPTHS = [0.25, 0.5, 0.75]


def get_mcq_questions() -> List[dict]:
    return MCQ_QUESTIONS


def get_genqa_questions() -> List[dict]:
    return GENQA_QUESTIONS


def build_divergence_prompts(context_lengths_words: List[int], per_length: int) -> List[dict]:
    """Deterministic prompts at approximate word-count targets."""
    prompts: List[dict] = []
    for length in context_lengths_words:
        for i in range(per_length):
            prompts.append(
                {
                    "name": f"div_w{length}_i{i}",
                    "target_words": length,
                    "text": _expand_to_length(length, seed=(length * 1000 + i)),
                }
            )
    return prompts


def _expand_to_length(target_words: int, seed: int) -> str:
    rng = random.Random(seed)
    sentences: List[str] = []
    for p in PASSAGES:
        sentences.extend(s.strip() for s in p.replace("\n", " ").split(". ") if s.strip())
    filler = list(FILLER_SENTENCES)
    rng.shuffle(filler)
    pool = sentences + filler
    out: List[str] = []
    count = 0
    idx = 0
    while count < target_words:
        s = pool[idx % len(pool)]
        idx += 1
        out.append(s if s.endswith(".") else s + ".")
        count += len(s.split())
    # Keep order stable per prompt so the text reads naturally; add a short
    # instruction tail so models have something coherent to continue.
    body = " ".join(out)
    tail = "\n\nContinue the text in the same style."
    return body + tail


def get_niah_cases(context_lengths_words: List[int]) -> List[dict]:
    """One needle per (context length, depth), deterministic values."""
    cases: List[dict] = []
    for length in context_lengths_words:
        for depth in NIAH_DEPTHS:
            value = f"ZEPHYR-{length}-{int(depth * 100)}"
            cases.append(_build_niah_case(length, depth, value))
    return cases


def _build_niah_case(context_words: int, depth: float, value: str) -> dict:
    rng = random.Random(hash((context_words, depth)) & 0xFFFFFFFF)
    sentences: List[str] = []
    count = 0
    while count < context_words:
        s = FILLER_SENTENCES[len(sentences) % len(FILLER_SENTENCES)]
        sentences.append(s)
        count += len(s.split())
    needle = f"One of the special codes written in these notes is {value}."
    pos = min(len(sentences) - 1, max(0, int(len(sentences) * depth)))
    sentences.insert(pos, needle)
    body = " ".join(sentences)
    prompt = (
        body
        + "\n\nQuestion: What is the special code written in these notes? "
        "Answer with only the code."
    )
    return {
        "id": f"niah_w{context_words}_d{int(depth * 100)}",
        "context_words": context_words,
        "depth": depth,
        "answer": value,
        "prompt": prompt,
    }
