"""In-process background job runner. Each comparison run is an asyncio task
whose status/progress/result land in the store as it goes."""
from __future__ import annotations

import asyncio
from typing import Coroutine, Dict

from . import store

_tasks: Dict[str, asyncio.Task] = {}


def start(run_id: str, coro: Coroutine) -> None:
    _tasks[run_id] = asyncio.create_task(_wrap(run_id, coro))


def cancel(run_id: str) -> bool:
    task = _tasks.get(run_id)
    if task and not task.done():
        task.cancel()
        return True
    return False


async def _wrap(run_id: str, coro: Coroutine) -> None:
    try:
        result = await coro
        store.update_run(
            run_id,
            status="completed",
            result=result,
            progress={"step": "done", "pct": 100},
        )
    except asyncio.CancelledError:
        store.update_run(run_id, status="cancelled")
        raise
    except Exception as e:  # noqa: BLE001 - surface anything to the UI
        store.update_run(
            run_id,
            status="failed",
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        _tasks.pop(run_id, None)
