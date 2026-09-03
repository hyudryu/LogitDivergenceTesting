const base = '/api'

async function req(path, opts = {}) {
  const res = await fetch(base + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ? (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)) : JSON.stringify(body)
    } catch {
      /* keep statusText */
    }
    throw new Error(detail)
  }
  if (res.status === 204) return null
  return res.json()
}

export const listSuites = () => req('/suites')
export const listRuns = () => req('/runs')
export const getRun = (id) => req(`/runs/${id}`)
export const deleteRun = (id) => req(`/runs/${id}`, { method: 'DELETE' })
export const createRun = (body) => req('/runs', { method: 'POST', body: JSON.stringify(body) })
