import React, { useState } from 'react'
import { createRun } from '../api.js'

const KV_PRESETS = ['bf16', 'fp8', 'fp8_e5m2', 'nvfp4', 'int4', 'auto']
const ALL_SUITES = ['divergence', 'mcq', 'niah', 'genqa']

const newEndpoint = (name, kv, baseline) => ({
  name,
  base_url: '',
  model: '',
  kv_dtype: kv,
  api_key: '',
  is_baseline: baseline,
  mock: false,
})

const fmtInts = (s) =>
  String(s)
    .split(',')
    .map((x) => parseInt(x.trim(), 10))
    .filter((n) => !Number.isNaN(n) && n > 0)

export default function NewRunForm({ onCreated }) {
  const [name, setName] = useState('')
  const [endpoints, setEndpoints] = useState([
    { ...newEndpoint('bf16-baseline', 'bf16', true), base_url: 'http://localhost:8001/v1' },
    { ...newEndpoint('fp8', 'fp8', false), base_url: 'http://localhost:8002/v1' },
  ])
  const [suites, setSuites] = useState(['divergence', 'mcq'])
  const [params, setParams] = useState({
    num_prompts: 8,
    max_new_tokens: 64,
    top_k_logprobs: 10,
    context_lengths_words: '128, 512, 2048',
    niah_contexts_words: '256, 1024',
    mcq_top_k: 20,
    bucket_size: 256,
  })
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const setEndpoint = (i, patch) =>
    setEndpoints((eps) => eps.map((e, j) => (j === i ? { ...e, ...patch } : e)))

  const setBaseline = (i) =>
    setEndpoints((eps) => eps.map((e, j) => ({ ...e, is_baseline: j === i })))

  const toggleSuite = (s) =>
    setSuites((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]))

  const submit = async (ev) => {
    ev.preventDefault()
    setError(null)
    const cleaned = endpoints.map((e) => ({
      name: e.name.trim(),
      base_url: e.mock ? `mock://${e.name || 'x'}` : e.base_url.trim(),
      model: e.model.trim() || null,
      kv_dtype: e.kv_dtype,
      api_key: e.api_key || null,
      backend: e.mock ? 'mock' : 'openai',
      is_baseline: e.is_baseline,
    }))
    if (cleaned.length < 2) return setError('Need at least two endpoints to compare.')
    if (cleaned.some((e) => !e.name)) return setError('Every endpoint needs a name.')
    if (!e_ok(cleaned)) return setError('Every real endpoint needs a base URL (e.g. http://localhost:8001/v1).')
    if (suites.length === 0) return setError('Select at least one suite.')
    setSubmitting(true)
    try {
      const body = {
        name: name.trim() || null,
        endpoints: cleaned,
        params: {
          suites,
          num_prompts: Number(params.num_prompts),
          max_new_tokens: Number(params.max_new_tokens),
          top_k_logprobs: Number(params.top_k_logprobs),
          context_lengths_words: fmtInts(params.context_lengths_words),
          niah_contexts_words: fmtInts(params.niah_contexts_words),
          mcq_top_k: Number(params.mcq_top_k),
          bucket_size: Number(params.bucket_size),
        },
      }
      const run = await createRun(body)
      onCreated(run.id)
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section>
      <h2>New comparison run</h2>
      <p className="muted">
        Point each row at the <strong>same model</strong> served with a different KV-cache dtype
        (e.g. vLLM <code>--kv-cache-dtype fp8</code> vs default BF16). The row marked ★ is the
        baseline everything else is compared against.
      </p>
      <form onSubmit={submit} className="newrun">
        <label className="field">
          <span>Run name (optional)</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Llama-8B bf16 vs fp8" />
        </label>

        <h3>Serving endpoints</h3>
        {endpoints.map((e, i) => (
          <div className="endpoint-row" key={i}>
            <label className="baseline-pick" title="Baseline">
              <input type="radio" checked={!!e.is_baseline} onChange={() => setBaseline(i)} />
              <span>★</span>
            </label>
            <label>
              <span>Name</span>
              <input value={e.name} onChange={(ev) => setEndpoint(i, { name: ev.target.value })} placeholder="fp8" />
            </label>
            <label className="grow">
              <span>Base URL (OpenAI-compatible)</span>
              <input
                value={e.base_url}
                disabled={e.mock}
                onChange={(ev) => setEndpoint(i, { base_url: ev.target.value })}
                placeholder="http://localhost:8002/v1"
              />
            </label>
            <label>
              <span>Model (optional)</span>
              <input value={e.model || ''} onChange={(ev) => setEndpoint(i, { model: ev.target.value })} placeholder="auto" />
            </label>
            <label>
              <span>KV dtype</span>
              <input
                list="kv-presets"
                value={e.kv_dtype}
                onChange={(ev) => setEndpoint(i, { kv_dtype: ev.target.value })}
              />
            </label>
            <label>
              <span>API key</span>
              <input
                type="password"
                value={e.api_key || ''}
                onChange={(ev) => setEndpoint(i, { api_key: ev.target.value })}
                placeholder="none"
              />
            </label>
            <label className="mock-check">
              <span>Mock</span>
              <input type="checkbox" checked={e.mock} onChange={(ev) => setEndpoint(i, { mock: ev.target.checked })} />
            </label>
            {endpoints.length > 2 && (
              <button
                type="button"
                className="ghost small"
                onClick={() => setEndpoints((eps) => eps.filter((_, j) => j !== i))}
              >
                ✕
              </button>
            )}
          </div>
        ))}
        <datalist id="kv-presets">
          {KV_PRESETS.map((k) => (
            <option key={k} value={k} />
          ))}
        </datalist>
        <button type="button" className="ghost" onClick={() => setEndpoints((eps) => [...eps, newEndpoint('', 'fp8', false)])}>
          + Add endpoint
        </button>

        <h3>Suites</h3>
        <div className="suite-row">
          {ALL_SUITES.map((s) => (
            <label key={s} className="suite-check">
              <input type="checkbox" checked={suites.includes(s)} onChange={() => toggleSuite(s)} />
              <span>
                <strong>{s}</strong>
              </span>
            </label>
          ))}
        </div>

        <h3>Parameters</h3>
        <div className="param-grid">
          <label>
            <span>Prompts per context length</span>
            <input type="number" min="1" max="64" value={params.num_prompts} onChange={(e) => setParams({ ...params, num_prompts: e.target.value })} />
          </label>
          <label>
            <span>Context lengths (words)</span>
            <input value={params.context_lengths_words} onChange={(e) => setParams({ ...params, context_lengths_words: e.target.value })} />
          </label>
          <label>
            <span>Max new tokens (generation)</span>
            <input type="number" min="1" max="512" value={params.max_new_tokens} onChange={(e) => setParams({ ...params, max_new_tokens: e.target.value })} />
          </label>
          <label>
            <span>Top-k logprobs (≤20)</span>
            <input type="number" min="1" max="20" value={params.top_k_logprobs} onChange={(e) => setParams({ ...params, top_k_logprobs: e.target.value })} />
          </label>
          <label>
            <span>Position bucket size</span>
            <input type="number" min="16" max="4096" value={params.bucket_size} onChange={(e) => setParams({ ...params, bucket_size: e.target.value })} />
          </label>
          <label>
            <span>NIAH context lengths (words)</span>
            <input value={params.niah_contexts_words} onChange={(e) => setParams({ ...params, niah_contexts_words: e.target.value })} />
          </label>
        </div>

        {error && <div className="error-box">{error}</div>}
        <div className="actions">
          <button className="primary" type="submit" disabled={submitting}>
            {submitting ? 'Starting…' : 'Start comparison'}
          </button>
        </div>
      </form>
    </section>
  )
}

function e_ok(cleaned) {
  return cleaned.every((e) => e.backend === 'mock' || /^https?:\/\//.test(e.base_url))
}
