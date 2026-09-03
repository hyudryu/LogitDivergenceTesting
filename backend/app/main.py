"""FastAPI application: create/list/inspect/delete comparison runs."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import jobs, store
from .datasets import SUITE_INFO
from .runner import run_comparison
from .schemas import RunCreate


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.init_db()
    yield


app = FastAPI(title="Logit Divergence Harness", version="0.1.0", lifespan=lifespan)

# Local dev tool: the Vite dev server (5173) talks to this API (8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/api/suites")
async def suites() -> Dict[str, str]:
    return SUITE_INFO


@app.post("/api/runs", status_code=201)
async def create_run(body: RunCreate) -> Dict[str, Any]:
    if not any(e.is_baseline for e in body.endpoints):
        body.endpoints[0].is_baseline = True
    name = body.name or f"{body.endpoints[0].name} baseline comparison"
    run_id = store.create_run(
        name=name,
        endpoints=[e.model_dump() for e in body.endpoints],
        params=body.params.model_dump(),
    )

    def progress(label: str, pct: int) -> None:
        store.update_run(run_id, progress={"step": label, "pct": pct})

    jobs.start(run_id, run_comparison(body.endpoints, body.params, progress))
    return {"id": run_id, "status": "queued"}


@app.get("/api/runs")
async def list_runs() -> Dict[str, Any]:
    return {"runs": store.list_runs()}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.delete("/api/runs/{run_id}", status_code=204)
async def delete_run(run_id: str) -> None:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    jobs.cancel(run_id)
    store.delete_run(run_id)
