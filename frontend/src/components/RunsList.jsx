import React, { useEffect, useState } from 'react'
import { deleteRun, listRuns } from '../api.js'

const STATUS_CLASS = {
  queued: 'pill queued',
  running: 'pill running',
  completed: 'pill completed',
  failed: 'pill failed',
  cancelled: 'pill cancelled',
}

export default function RunsList({ onOpen, onNew }) {
  const [runs, setRuns] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    const load = () =>
      listRuns()
        .then((r) => alive && setRuns(r.runs))
        .catch((e) => alive && setError(e.message))
    load()
    const t = setInterval(load, 3000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [])

  const remove = async (id) => {
    await deleteRun(id)
    setRuns((rs) => rs.filter((r) => r.id !== id))
  }

  if (error) return <div className="error-box">Failed to load runs: {error}</div>
  if (!runs) return <div className="muted">Loading…</div>

  return (
    <section>
      <div className="section-head">
        <h2>Comparison runs</h2>
        <button className="primary" onClick={onNew}>
          + New comparison
        </button>
      </div>
      {runs.length === 0 ? (
        <div className="empty-state">
          <p>No runs yet. Create a comparison between two serving configs of the same model.</p>
          <p className="muted">
            Tip: you can try the UI without GPUs by adding two <em>mock</em> endpoints
            (e.g. <code>bf16</code> and <code>nvfp4</code>) in the new-comparison form.
          </p>
          <button className="primary" onClick={onNew}>
            Create your first run
          </button>
        </div>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Configs</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Created</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id} className="clickable" onClick={() => onOpen(r.id)}>
                <td>{r.name}</td>
                <td>
                  {(r.endpoints || []).map((e, i) => (
                    <span key={i} className="chip">
                      {e.name} · {e.kv_dtype}
                      {e.is_baseline ? ' ★' : ''}
                    </span>
                  ))}
                </td>
                <td>
                  <span className={STATUS_CLASS[r.status] || 'pill'}>{r.status}</span>
                </td>
                <td className="muted">{r.progress?.step || ''}</td>
                <td className="muted">{new Date(r.created_at * 1000).toLocaleString()}</td>
                <td>
                  <button
                    className="ghost small"
                    onClick={(ev) => {
                      ev.stopPropagation()
                      remove(r.id)
                    }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}
