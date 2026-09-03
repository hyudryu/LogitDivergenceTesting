import React, { useState } from 'react'
import RunsList from './components/RunsList.jsx'
import NewRunForm from './components/NewRunForm.jsx'
import RunDetail from './components/RunDetail.jsx'

export default function App() {
  const [view, setView] = useState({ name: 'runs' })

  return (
    <div className="app">
      <header className="topbar">
        <h1 className="brand" onClick={() => setView({ name: 'runs' })}>
          Logit Divergence Harness
        </h1>
        <span className="tagline">KV-cache precision: BF16 vs FP8 vs NVFP4</span>
        <nav>
          <button
            className={view.name === 'runs' ? 'active' : ''}
            onClick={() => setView({ name: 'runs' })}
          >
            Runs
          </button>
          <button
            className={view.name === 'new' ? 'active' : ''}
            onClick={() => setView({ name: 'new' })}
          >
            + New comparison
          </button>
        </nav>
      </header>
      <main>
        {view.name === 'runs' && (
          <RunsList onOpen={(id) => setView({ name: 'detail', id })} onNew={() => setView({ name: 'new' })} />
        )}
        {view.name === 'new' && <NewRunForm onCreated={(id) => setView({ name: 'detail', id })} />}
        {view.name === 'detail' && <RunDetail id={view.id} onBack={() => setView({ name: 'runs' })} />}
      </main>
    </div>
  )
}
