import asyncio

from app.runner import run_comparison
from app.schemas import EndpointConfig, RunParams


def _mock_endpoints():
    return [
        EndpointConfig(
            name="bf16",
            base_url="mock://bf16",
            backend="mock",
            kv_dtype="bf16",
            is_baseline=True,
        ),
        EndpointConfig(
            name="fp8",
            base_url="mock://fp8",
            backend="mock",
            kv_dtype="fp8",
        ),
    ]


def test_mock_end_to_end():
    params = RunParams(
        suites=["all"],
        num_prompts=2,
        context_lengths_words=[64, 128],
        niah_contexts_words=[64],
        max_new_tokens=16,
        bucket_size=64,
    )
    progress = []
    result = asyncio.run(run_comparison(_mock_endpoints(), params, lambda l, p: progress.append((l, p))))

    assert set(result["accuracy"].keys()) == {"bf16", "fp8"}
    assert result["baseline"] == "bf16"
    assert result["prompt_logprobs_support"] == {"bf16": True, "fp8": True}

    pair = result["pairs"]["bf16 vs fp8"]
    tf = pair["teacher_forced"]
    assert tf["supported"] is True
    assert tf["n_positions"] > 0
    assert tf["abs_dlogprob_by_position"]
    assert tf["top1_agreement_by_position"]

    gen = pair["generation"]
    assert gen["n_prompts"] >= 1
    assert gen["mean_first_divergence_token"] >= 0

    # Mock fp8 noise must produce measurable divergence.
    assert tf["mean_abs_dlogprob"] > 0
    assert gen["exact_match_rate"] < 1.0


def test_mock_identical_configs_diverge_negligibly():
    eps = [
        EndpointConfig(name="bf16a", base_url="mock://a", backend="mock", kv_dtype="bf16", is_baseline=True),
        EndpointConfig(name="bf16b", base_url="mock://b", backend="mock", kv_dtype="bf16"),
    ]
    params = RunParams(
        suites=["divergence"],
        num_prompts=1,
        context_lengths_words=[64],
        max_new_tokens=8,
    )
    result = asyncio.run(run_comparison(eps, params, lambda l, p: None))
    pair = result["pairs"]["bf16a vs bf16b"]
    assert pair["teacher_forced"]["supported"]
