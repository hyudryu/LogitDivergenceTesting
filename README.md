# Logit Divergence Testing

A local harness for measuring how much a model's output distribution drifts when you
change the **KV-cache precision** it is served with (BF16 → FP8 → NVFP4), and whether
that drift actually costs you accuracy.

One React dashboard, one FastAPI backend, and any number of OpenAI-compatible LLM
servers (vLLM, llama.cpp server, LM Studio, …) that you point it at.

```
┌────────────┐   /api    ┌───────────────┐   logprobs   ┌──────────────────────┐
│ React UI   │ ────────► │ FastAPI       │ ───────────► │ server A (bf16 KV)   │
│ (Vite)     │           │ harness       │ ───────────► │ server B (fp8 KV)    │
└────────────┘           └───────────────┘              │ server C (nvfp4 KV)  │
                                                        └──────────────────────┘
```

---

## Methodology

The core question: *same weights, same prompt, only the KV-cache dtype differs — how
far apart do the logits get, where does the gap grow, and does it change the answers?*
The harness answers that with three layers, from most sensitive to most human-meaningful.

### Layer 1 — Teacher-forced logprob scan (the sensitive detector)

Both serving configs receive **exactly the same prompt** (input tokens are forced), so
any difference in the per-position next-token distributions is *pure KV-cache signal* —
there is no generation drift confounding it. Using vLLM's `prompt_logprobs` extension:

* `prompt_logprobs=0` → the logprob each config assigns to the **realized** token at
  every position. The per-position difference forms the **Δlogprob-vs-depth curve**.
  KV quantization error compounds as attention reaches further back, so this curve
  rising with position is the classic accumulation signature.
* `prompt_logprobs=k` → each config's **top-k next-token distribution** at every
  position, compared with:
  * **top-1 agreement** — how often both configs pick the same argmax token
  * **Jensen-Shannon divergence** (nats) between the renormalized top-k sets
  * **entropy delta** — does the degraded cache make the model more or less confident

Requires `prompt_logprobs` (vLLM has it; llama.cpp generally does not). The backend
probes capability and falls back to Layer 2 when unavailable.

### Layer 2 — Greedy generation divergence (the amplifier)

`temperature=0` completions on both configs. Tiny distribution shifts that Layer 1
catches may or may not survive argmax; this layer measures what actually happens to
decoded text:

* **first-divergence token** — how many tokens until the two configs disagree
* **per-position top-1 agreement** across the generated sequence
* **exact-match rate** — fraction of prompts with token-identical generations

### Layer 3 — Task accuracy (does it matter?)

Run once per config so any pair is comparable:

| Suite | What it does | Scoring |
|-------|--------------|---------|
| `mcq` | Multiple-choice questions; score each answer letter by its next-token logprob (lm-eval-harness style) | argmax over A/B/C/D logprobs |
| `niah` | Synthetic needle-in-a-haystack at controlled context lengths × needle depths | substring match on the needle |
| `genqa` | Short-form QA + arithmetic, greedy decoding | normalized exact / numeric match |

Long-context recall (`niah`) is where you should expect KV quantization to bite first.

### Experimental controls (important)

1. **Hold everything except KV dtype constant.** Same checkpoint on both servers — if
   you also switch weight quantization (e.g. an FP8-weights checkpoint vs NVFP4-weights
   checkpoint), you are measuring *both* effects at once. For pure KV comparisons,
   toggle only the server's KV flag (see commands below).
2. **Greedy decoding everywhere** (`temperature=0`). Sampled generations add noise that
   dwarfs cache-precision effects.
3. **Same tokenizer on both sides.** Metrics key tokens by decoded string; that mapping
   is only 1:1 within one tokenizer, which is why a comparison run must use one model.
4. **Top-k truncation caveat.** Top-k JS is computed after flooring missing tokens and
   renormalizing over the union. It has a small bias, but it is the *same* bias for
   every config, so relative comparisons remain valid. Full-vocab KL would require
   serving-side logit hooks; it is a good future extension.

### Reading the results

* `mean |Δ logprob|` ≈ `1e-4`–`1e-3` with top-1 agreement ≈ 100% → noise-level drift.
* Rising Δlogprob/JS curves vs position → accumulating cache error (worse with longer
  contexts than short ones).
* NIAH accuracy dropping at depth before MCQ/GenQA move → retrieval is the canary.
* Diverging generations with identical accuracy → drift is real but harmless for the
  task; trust Layer 1/2 trends plus Layer 3 together, not any single number.

---

## Quickstart (no GPUs needed)

Mock endpoints synthesize dtype-scaled noise so you can exercise the whole pipeline:

```bash
# terminal 1 — backend
cd backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run.py            # http://127.0.0.1:8000

# terminal 2 — frontend
cd frontend
npm install
npm run dev                            # http://localhost:5173
```

In the UI: **+ New comparison** → tick **Mock** on both endpoint rows → set KV dtypes
to `bf16` (baseline) and `nvfp4` → suites `divergence` + `mcq` → Start.

## Real servers

Run the same model twice with only the KV dtype changed, then enter both endpoints:

```bash
# vLLM: baseline (bf16/auto KV)
vllm serve <model> --port 8001

# vLLM: fp8 KV cache
vllm serve <model> --kv-cache-dtype fp8 --port 8002
```

Check `vllm serve --help` for the `--kv-cache-dtype` values your version supports
(`fp8`, `fp8_e5m2`, and on newer builds FP4/NVFP4 KV cache on Blackwell). llama.cpp:

```bash
llama-server -m model.gguf -c 8192 --port 8001              # f16 KV (baseline)
llama-server -m model.gguf -c 8192 --port 8002 -ctk q8_0 -ctv q8_0
```

Notes:
* llama.cpp has no `prompt_logprobs`; runs against it use generation-mode divergence
  plus the accuracy suites (the UI shows a warning).
* For best sensitivity keep prompts reasonably long — that is what `context_lengths_words`
  controls for the divergence suite.
* API keys are optional; if set they are kept in memory for the run and are not stored
  in the database (only a flag-free copy of the endpoint config is persisted).

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/health` | liveness |
| GET  | `/api/suites` | suite descriptions |
| POST | `/api/runs` | create + start a comparison (`{name?, endpoints[≥2], params}`) |
| GET  | `/api/runs` | list runs |
| GET  | `/api/runs/{id}` | run detail incl. result JSON |
| DELETE | `/api/runs/{id}` | cancel (if running) + delete |

`params`: `suites` (subset of `divergence|mcq|niah|genqa|all`), `num_prompts`,
`max_new_tokens`, `top_k_logprobs` (≤20), `context_lengths_words`, `niah_contexts_words`,
`mcq_top_k`, `bucket_size`.

Runs are stored in `backend/data/harness.db` (override with `LOGIT_DB_PATH`).

## Tests

```bash
cd backend
.venv\Scripts\python -m pytest tests -q
```

## Limitations / future work

* Full-vocab KL via serving-side logit hooks (vLLM custom worker / logits processor).
* Continuous divergence-vs-depth plots per individual prompt, not just pooled buckets.
* MTEB-style embedding drift and speculative-decoding acceptance-rate impact as
  additional downstream signals.
* Automatic multi-round sweeps (dtype × context length × model grid).
