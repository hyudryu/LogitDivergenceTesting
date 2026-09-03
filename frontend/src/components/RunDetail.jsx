import React, { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getRun } from '../api.js'

const COLORS = ['#4fc3f7', '#ffb74d', '#81c784', '#e57373', '#ba68c8', '#fff176']

const fmt = (v, digits = 4) => (v === null || v === undefined ? '—' : Number(v).toPrecision(digits))
const pct = (v) => (v === null || v === undefined ? '—' : `${(100 * v).toFixed(1)}%`)

function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  )
}

export default function RunDetail({ id, onBack }) {
  const [run, setRun] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    let timer = null
    const load = async () => {
      try {
        const r = await getRun(id)
        if (!alive) return
        setRun(r)
        setError(null)
        if (r.status === 'running' || r.status === 'queued') {
          timer = setTimeout(load, 2000)
        }
      } catch (e) {
        if (alive) setError(e.message)
      }
    }
    load()
    return () => {
      alive = false
      if (timer) clearTimeout(timer)
    }
  }, [id])

  if (error) return <div className="error-box">Failed to load run: {error}</div>
  if (!run) return <div className="muted">Loading…</div>

  const { result } = run

  return (
    <section>
      <div className="section-head">
        <div>
          <button className="ghost small" onClick={onBack}>
            ← All runs
          </button>
          <h2>
            {run.name} <span className={`pill ${run.status}`}>{run.status}</span>
          </h2>
        </div>
      </div>

      {(run.status === 'running' || run.status === 'queued') && (
        <div className="progress-box">
          <div className="progress-bar">
            <div style={{ width: `${run.progress?.pct || 0}%` }} />
          </div>
          <div className="muted">{run.progress?.step || 'working…'}</div>
        </div>
      )}
      {run.status === 'failed' && <div className="error-box">{run.error}</div>}

      {result && (
        <>
          <ConfigSummary configs={result.configs || []} baseline={result.baseline} />

          {result.accuracy && <AccuracySection configs={result.configs || []} accuracy={result.accuracy} />}

          {result.pairs &&
            Object.entries(result.pairs).map(([key, pair]) => (
              <PairSection key={key} pairKey={key} pair={pair} />
            ))}
        </>
      )}
    </section>
  )
}

function ConfigSummary({ configs, baseline }) {
  return (
    <div className="card">
      <h3>Configs</h3>
      <table className="table">
        <thead>
          <tr>
            <th>Name</th>
            <th>KV dtype</th>
            <th>Model</th>
            <th>Endpoint</th>
            <th>prompt_logprobs</th>
          </tr>
        </thead>
        <tbody>
          {configs.map((c) => (
            <tr key={c.name}>
              <td>
                {c.name}
                {c.name === baseline ? ' ★' : ''}
              </td>
              <td>
                <span className="chip">{c.kv_dtype}</span>
              </td>
              <td className="muted">{c.resolved_model || c.model || '—'}</td>
              <td className="muted">{c.mock ? '(mock)' : c.base_url}</td>
              <td>{c.prompt_logprobs ? '✓' : '✗ (generation mode only)'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function AccuracySection({ configs, accuracy }) {
  const suites = Object.keys(accuracy[configs[0]?.name] || {})
  if (!suites.length) return null
  const data = suites.map((s) => {
    const row = { suite: s }
    for (const c of configs) {
      row[c.name] = accuracy[c.name]?.[s]?.accuracy ?? 0
    }
    return row
  })
  return (
    <div className="card">
      <h3>Task accuracy by config</h3>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
            <XAxis dataKey="suite" stroke="#8b93a7" />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${(100 * v).toFixed(0)}%`} stroke="#8b93a7" />
            <Tooltip formatter={(v) => pct(v)} />
            <Legend />
            {configs.map((c, i) => (
              <Bar key={c.name} dataKey={c.name} fill={COLORS[i % COLORS.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>Suite</th>
            {configs.map((c) => (
              <th key={c.name}>
                {c.name}
                {c.is_baseline ? ' ★' : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {suites.map((s) => (
            <tr key={s}>
              <td>{s}</td>
              {configs.map((c) => {
                const a = accuracy[c.name]?.[s]
                return (
                  <td key={c.name}>
                    {pct(a?.accuracy)} <span className="muted">({a?.n ?? 0})</span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PairSection({ pairKey, pair }) {
  const tf = pair.teacher_forced || {}
  const gen = pair.generation || {}
  return (
    <div className="card">
      <h3>Divergence: {pairKey}</h3>

      <div className="stats-row">
        <Stat label="mean |Δ logprob| (teacher-forced)" value={fmt(tf.mean_abs_dlogprob)} />
        <Stat label="p99 |Δ logprob|" value={fmt(tf.p99_abs_dlogprob)} />
        <Stat label="top-1 agreement (scan)" value={pct(tf.top1_agreement)} />
        <Stat label="mean top-k JS (scan)" value={fmt(tf.mean_js_topk)} />
        <Stat label="identical generations" value={pct(gen.exact_match_rate)} />
        <Stat label="1st divergent token (mean)" value={fmt(gen.mean_first_divergence_token, 3)} />
      </div>

      {!tf.supported && (
        <div className="warn-box">
          prompt_logprobs is not supported by these endpoints — teacher-forced curves are
          unavailable (vLLM exposes it; most other servers do not). Generation-based metrics
          below still work everywhere.
        </div>
      )}

      {tf.supported && tf.abs_dlogprob_by_position?.length > 0 && (
        <ChartBlock
          title="Realized-token |Δ logprob| vs context position (bucketed)"
          hint="KV quantization error accumulates with depth; a rising curve means the further into the context, the larger the logit drift."
          data={tf.abs_dlogprob_by_position}
          xKey="bucket"
          lines={[{ key: 'mean', name: 'mean |Δ logprob|' }]}
        />
      )}

      {tf.supported && tf.top1_agreement_by_position?.length > 0 && (
        <ChartBlock
          title="Top-1 agreement & top-k JS vs context position"
          hint="Agreement: do both configs pick the same argmax token? JS: divergence between renormalized top-k distributions."
          data={mergeByBucket(tf.top1_agreement_by_position, tf.js_by_position)}
          xKey="bucket"
          lines={[
            { key: 'agreement', name: 'top-1 agreement' },
            { key: 'js', name: 'JS divergence' },
          ]}
          pctKey="agreement"
        />
      )}

      {gen.top1_agreement_by_position?.length > 0 && (
        <ChartBlock
          title="Greedy generation: top-1 agreement by generated token index"
          hint="How quickly tiny distribution shifts flip the argmax once the model starts generating freely."
          data={gen.top1_agreement_by_position.map((d) => ({ ...d, mean: d.mean }))}
          xKey="bucket"
          lines={[{ key: 'mean', name: 'top-1 agreement' }]}
          pctKey="mean"
        />
      )}
    </div>
  )
}

function mergeByBucket(a, b) {
  const out = new Map()
  for (const d of a || []) out.set(d.bucket, { bucket: d.bucket, agreement: d.mean })
  for (const d of b || []) {
    const cur = out.get(d.bucket) || { bucket: d.bucket }
    cur.js = d.mean
    out.set(d.bucket, cur)
  }
  return [...out.values()].sort((x, y) => x.bucket - y.bucket)
}

function ChartBlock({ title, hint, data, xKey, lines, pctKey }) {
  return (
    <div className="chart-block">
      <h4>{title}</h4>
      {hint && <p className="muted small-text">{hint}</p>}
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
            <XAxis dataKey={xKey} stroke="#8b93a7" />
            <YAxis stroke="#8b93a7" domain={pctKey ? [0, 1] : ['auto', 'auto']} />
            <Tooltip />
            <Legend />
            {lines.map((l, i) => (
              <Line
                key={l.key}
                type="monotone"
                dataKey={l.key}
                name={l.name}
                stroke={COLORS[i % COLORS.length]}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
