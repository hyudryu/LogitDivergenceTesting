from app.datasets import (
    build_divergence_prompts,
    get_genqa_questions,
    get_mcq_questions,
    get_niah_cases,
)


def test_divergence_prompts_deterministic_and_sized():
    a = build_divergence_prompts([128, 512], 2)
    b = build_divergence_prompts([128, 512], 2)
    assert a == b
    assert len(a) == 4
    for p in a:
        words = len(p["text"].split())
        assert words >= p["target_words"]
        assert words < p["target_words"] * 1.5 + 40


def test_mcq_questions_well_formed():
    for q in get_mcq_questions():
        assert q["answer"] in ("A", "B", "C", "D")
        assert set(q["options"].keys()) == {"A", "B", "C", "D"}


def test_genqa_questions_have_answers():
    for q in get_genqa_questions():
        assert q["answer"].strip()


def test_niah_needle_is_embedded_and_findable():
    cases = get_niah_cases([128, 256])
    assert len(cases) == 6  # 2 lengths x 3 depths
    for case in cases:
        assert case["answer"].lower() in case["prompt"].lower()
