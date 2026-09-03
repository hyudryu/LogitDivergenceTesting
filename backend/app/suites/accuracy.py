"""Downstream accuracy suites. Each is run once per serving config so any
config can be compared against the baseline afterward:

  mcq   - multiple choice: pick the option letter whose next-token logprob is
          highest (the standard lm-eval-harness style logprob scoring).
  niah  - needle-in-a-haystack with a synthetic needle placed at controlled
          depths and context lengths; substring match on the needle value.
  genqa - short-form QA, greedy decoding, normalized exact/numeric match.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, List

from ..datasets import get_genqa_questions, get_mcq_questions, get_niah_cases
from ..llm_client import BaseClient
from ..schemas import RunParams

_LETTER_STRIP = " .,:;-'\"`()"


async def run_suite(
    kind: str,
    client: BaseClient,
    params: RunParams,
    on_progress: Callable[[str], None],
) -> dict:
    if kind == "mcq":
        return await run_mcq(client, params, on_progress)
    if kind == "niah":
        return await run_niah(client, params, on_progress)
    if kind == "genqa":
        return await run_genqa(client, params, on_progress)
    raise ValueError(f"unknown accuracy suite: {kind}")


def _mcq_prompt(q: dict) -> str:
    lines = [q["question"], ""]
    for letter in ("A", "B", "C", "D"):
        lines.append(f"{letter}. {q['options'][letter]}")
    lines += ["", "Answer:"]
    return "\n".join(lines)


async def run_mcq(client: BaseClient, params: RunParams, on_progress: Callable[[str], None]) -> dict:
    questions = get_mcq_questions()
    details: List[dict] = []
    for i, q in enumerate(questions):
        on_progress(f"mcq {i + 1}/{len(questions)}")
        topk = await client.next_token_topk(_mcq_prompt(q), params.mcq_top_k)
        letters: Dict[str, float] = {}
        for tok, lp in (topk or {}).items():
            s = tok.strip(_LETTER_STRIP).upper()
            if s in ("A", "B", "C", "D"):
                letters[s] = max(letters.get(s, -1e30), lp)
        pred = max(letters, key=letters.get) if letters else None
        details.append(
            {
                "id": q["id"],
                "expected": q["answer"],
                "predicted": pred,
                "correct": pred == q["answer"],
                "letter_logprobs": letters or None,
            }
        )
    correct = sum(1 for d in details if d["correct"])
    return {
        "accuracy": correct / len(details) if details else None,
        "n": len(details),
        "details": details,
    }


async def run_niah(client: BaseClient, params: RunParams, on_progress: Callable[[str], None]) -> dict:
    cases = get_niah_cases(params.niah_contexts_words)
    details: List[dict] = []
    by_ctx: Dict[int, List[bool]] = {}
    for i, case in enumerate(cases):
        on_progress(f"niah {i + 1}/{len(cases)} ({case['context_words']}w d{case['depth']})")
        r = await client.complete(case["prompt"], max_tokens=16)
        got = (r.text or "").lower()
        ok = case["answer"].lower() in got
        details.append(
            {
                "id": case["id"],
                "context_words": case["context_words"],
                "depth": case["depth"],
                "expected": case["answer"],
                "correct": ok,
            }
        )
        by_ctx.setdefault(case["context_words"], []).append(ok)
    return {
        "accuracy": sum(sum(v) for v in by_ctx.values()) / len(details) if details else None,
        "n": len(details),
        "by_context": {
            str(ctx): {"accuracy": sum(v) / len(v), "n": len(v)} for ctx, v in sorted(by_ctx.items())
        },
        "details": details,
    }


def _qa_prompt(q: dict) -> str:
    return f"Question: {q['question']}\nAnswer:"


_PUNCT_KEEP = re.compile(r"[^a-z0-9.,\- ]+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SPACES = re.compile(r"\s+")


def _normalize(text: str) -> str:
    s = text.strip().splitlines()[0] if text.strip() else ""
    s = s.lower()
    s = _PUNCT_KEEP.sub("", s)
    s = _ARTICLES.sub("", s)
    s = _SPACES.sub(" ", s).strip()
    return s


def _as_number(s: str):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def exact_match(expected: str, generated: str) -> bool:
    exp, got = _normalize(expected), _normalize(generated)
    if not got:
        return False
    if exp == got:
        return True
    n_exp, n_got = _as_number(exp), _as_number(got)
    if n_exp is not None and n_got is not None:
        return math.isclose(n_exp, n_got, rel_tol=1e-6, abs_tol=1e-9)
    return False


async def run_genqa(client: BaseClient, params: RunParams, on_progress: Callable[[str], None]) -> dict:
    questions = get_genqa_questions()
    details: List[dict] = []
    for i, q in enumerate(questions):
        on_progress(f"genqa {i + 1}/{len(questions)}")
        r = await client.complete(_qa_prompt(q), max_tokens=48)
        ok = exact_match(q["answer"], r.text)
        details.append(
            {
                "id": q["id"],
                "expected": q["answer"],
                "generated": (r.text or "")[:160],
                "correct": ok,
            }
        )
    correct = sum(1 for d in details if d["correct"])
    return {
        "accuracy": correct / len(details) if details else None,
        "n": len(details),
        "details": details,
    }
