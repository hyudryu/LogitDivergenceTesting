"""Run orchestration: resolves clients, probes capabilities, executes the
selected accuracy suites once per config and the divergence suite pairwise
against the baseline, reporting progress along the way."""
from __future__ import annotations

from typing import Callable, List

from .llm_client import BaseClient, make_client
from .schemas import EndpointConfig, RunParams
from .suites import accuracy, divergence


async def run_comparison(
    endpoints: List[EndpointConfig],
    params: RunParams,
    on_progress: Callable[[str, int], None],
) -> dict:
    names = [e.name for e in endpoints]
    if len(set(names)) != len(names):
        raise ValueError("Endpoint names must be unique")

    clients: dict[str, BaseClient] = {e.name: make_client(e) for e in endpoints}
    try:
        baseline = next((e for e in endpoints if e.is_baseline), endpoints[0])
        others = [e for e in endpoints if e.name != baseline.name]
        suites = params.expanded_suites()
        acc_suites = [s for s in suites if s != "divergence"]

        units = len(endpoints) * len(acc_suites) + len(others)
        unit_pct = 100.0 / units if units else 100.0
        state = {"units_done": 0}

        def make_cb(prefix: str):
            def cb(label: str) -> None:
                pct = min(99, int(unit_pct * state["units_done"]))
                on_progress(f"{prefix}: {label}", pct)

            return cb

        result: dict = {
            "baseline": baseline.name,
            "configs": [],
            "accuracy": {},
            "pairs": {},
        }

        # Resolve models (also validates reachability) and capabilities.
        for e in endpoints:
            await clients[e.name].resolve_model()
        tf_support = {e.name: await clients[e.name].probe_prompt_logprobs() for e in endpoints}

        result["configs"] = [
            {
                **e.model_dump(exclude={"api_key"}),
                "resolved_model": clients[e.name].model_id,
                "prompt_logprobs": tf_support[e.name],
            }
            for e in endpoints
        ]
        result["prompt_logprobs_support"] = tf_support

        # Accuracy suites: once per config (absolute numbers).
        for e in endpoints:
            result["accuracy"][e.name] = {}
            for s in acc_suites:
                cb = make_cb(f"{e.name}/{s}")
                result["accuracy"][e.name][s] = await accuracy.run_suite(
                    s, clients[e.name], params, cb
                )
                state["units_done"] += 1

        # Divergence: baseline vs every other config.
        for other in others:
            pair_key = f"{baseline.name} vs {other.name}"
            cb = make_cb(pair_key)
            result["pairs"][pair_key] = await divergence.run_divergence(
                clients[baseline.name],
                clients[other.name],
                baseline.name,
                other.name,
                params,
                tf_supported=tf_support.get(baseline.name) and tf_support.get(other.name),
                on_progress=cb,
            )
            state["units_done"] += 1

        return result
    finally:
        for c in clients.values():
            await c.close()
