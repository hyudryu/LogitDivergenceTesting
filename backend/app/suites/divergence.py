"""Pairwise divergence suite: compares a baseline serving config against a
degraded-KV config on identical inputs.

Layer 1 - teacher-forced scan (requires vLLM-style prompt_logprobs):
    Both servers receive the exact same prompt. Because the input tokens are
    forced, any difference in per-position logprobs comes purely from KV cache
    precision, not from generation drift. We collect, per prompt position:
      * realized-token logprob  (prompt_logprobs=0)  -> dlogprob curve vs depth
      * top-k distribution      (prompt_logprobs=k)  -> top-1 agreement + JS
    KV quantization error accumulates with context depth, so everything is
    bucketed by position.

Layer 2 - greedy generation:
    temperature=0 completions on both sides. Measures how small distribution
    shifts amplify into divergent decodings: first-divergence token index,
    per-position top-1 agreement, and exact token-sequence match rate.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from ..datasets import build_divergence_prompts
from ..llm_client import BaseClient
from ..metrics import bucket_series, compare_topk
from ..schemas import RunParams

# Cap generation-mode prompts: it is the slowest per-token part.
GEN_PROMPT_CAP = 6


async def run_divergence(
    client_a: BaseClient,
    client_b: BaseClient,
    name_a: str,
    name_b: str,
    params: RunParams,
    tf_supported: bool,
    on_progress: Callable[[str], None],
) -> dict:
    prompts = build_divergence_prompts(params.context_lengths_words, params.num_prompts)
    out: dict = {
        "baseline": name_a,
        "comparison": name_b,
        "teacher_forced": {"supported": False},
        "generation": {},
    }
    if tf_supported:
        out["teacher_forced"] = await _teacher_forced(client_a, client_b, prompts, params, on_progress)
    out["generation"] = await _generation(client_a, client_b, prompts[:GEN_PROMPT_CAP], params, on_progress)
    return out


async def _scan_one(client: BaseClient, prompt: str, k: int):
    r0 = await client.complete(prompt, max_tokens=1, prompt_logprobs=0)
    rk = await client.complete(prompt, max_tokens=1, prompt_logprobs=k)
    # Position 0 has no context and is always None; the scan loop skips it.
    realized = [e["logprob"] if e else None for e in (r0.prompt_scan or [])]
    tops = [e["topk"] if e else None for e in (rk.prompt_scan or [])]
    return realized, tops


async def _teacher_forced(
    a: BaseClient,
    b: BaseClient,
    prompts: List[dict],
    params: RunParams,
    on_progress: Callable[[str], None],
) -> dict:
    pos_dlp: List[int] = []
    val_dlp: List[float] = []
    val_signed: List[float] = []
    pos_dist: List[int] = []
    val_agree: List[float] = []
    val_js: List[float] = []
    prompt_summaries: List[dict] = []

    for i, p in enumerate(prompts):
        on_progress(f"teacher-forced scan {i + 1}/{len(prompts)} ({p['target_words']}w)")
        ra, ta = await _scan_one(a, p["text"], params.top_k_logprobs)
        rb, tb = await _scan_one(b, p["text"], params.top_k_logprobs)
        n = min(len(ra), len(rb))
        prompt_dlps: List[float] = []
        for pos in range(1, n):
            la_, lb_ = ra[pos], rb[pos]
            if la_ is None or lb_ is None:
                continue
            d = la_ - lb_
            prompt_dlps.append(d)
            pos_dlp.append(pos)
            val_dlp.append(abs(d))
            val_signed.append(d)
            ka, kb = ta[pos], tb[pos]
            if ka and kb:
                m = compare_topk(ka, kb)
                if m:
                    pos_dist.append(pos)
                    val_agree.append(1.0 if m["top1_agree"] else 0.0)
                    val_js.append(m["js"])
        if prompt_dlps:
            arr = np.array(prompt_dlps)
            prompt_summaries.append(
                {
                    "name": p["name"],
                    "approx_tokens": n,
                    "mean_abs_dlogprob": float(np.mean(np.abs(arr))),
                    "max_abs_dlogprob": float(np.max(np.abs(arr))),
                    "mean_signed_dlogprob": float(np.mean(arr)),
                }
            )

    def mean(xs: List[float]) -> Optional[float]:
        return float(np.mean(xs)) if xs else None

    return {
        "supported": True,
        "n_prompts": len(prompts),
        "n_positions": len(val_dlp),
        "mean_abs_dlogprob": mean(val_dlp),
        "mean_signed_dlogprob": mean(val_signed),
        "p99_abs_dlogprob": float(np.percentile(val_dlp, 99)) if val_dlp else None,
        "top1_agreement": mean(val_agree),
        "mean_js_topk": mean(val_js),
        "abs_dlogprob_by_position": bucket_series(pos_dlp, val_dlp, params.bucket_size),
        "top1_agreement_by_position": bucket_series(pos_dist, val_agree, params.bucket_size),
        "js_by_position": bucket_series(pos_dist, val_js, params.bucket_size),
        "prompts": prompt_summaries,
    }


async def _generation(
    a: BaseClient,
    b: BaseClient,
    prompts: List[dict],
    params: RunParams,
    on_progress: Callable[[str], None],
) -> dict:
    exact = 0
    first_divs: List[int] = []
    pos_rows: List[int] = []
    agree_rows: List[float] = []
    js_rows: List[float] = []
    per_prompt: List[dict] = []

    for i, p in enumerate(prompts):
        on_progress(f"greedy generation {i + 1}/{len(prompts)} ({p['target_words']}w)")
        ra = await a.complete(p["text"], max_tokens=params.max_new_tokens, top_k=params.top_k_logprobs)
        rb = await b.complete(p["text"], max_tokens=params.max_new_tokens, top_k=params.top_k_logprobs)
        toks_a = [e["token"] for e in ra.token_logprobs]
        toks_b = [e["token"] for e in rb.token_logprobs]
        identical = toks_a == toks_b
        exact += int(identical)
        first_div = None
        for j in range(min(len(toks_a), len(toks_b))):
            same = toks_a[j] == toks_b[j]
            pos_rows.append(j)
            agree_rows.append(1.0 if same else 0.0)
            if not same and first_div is None:
                first_div = j
            ea = ra.token_logprobs[j]["topk"]
            eb = rb.token_logprobs[j]["topk"]
            m = compare_topk(ea, eb)
            if m:
                js_rows.append(m["js"])
        if first_div is None:
            first_div = min(len(toks_a), len(toks_b))
        first_divs.append(first_div)
        per_prompt.append(
            {"name": p["name"], "identical": identical, "first_divergence_token": first_div}
        )

    return {
        "n_prompts": len(prompts),
        "max_new_tokens": params.max_new_tokens,
        "exact_match_rate": exact / len(prompts) if prompts else None,
        "mean_first_divergence_token": float(np.mean(first_divs)) if first_divs else None,
        "top1_agreement": float(np.mean(agree_rows)) if agree_rows else None,
        "mean_js_topk": float(np.mean(js_rows)) if js_rows else None,
        "top1_agreement_by_position": bucket_series(pos_rows, agree_rows, 1),
        "prompts": per_prompt,
    }
