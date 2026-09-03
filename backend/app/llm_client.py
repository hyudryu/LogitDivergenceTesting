"""Clients for OpenAI-compatible LLM servers (vLLM, llama.cpp server, LM Studio).

The harness needs token-level logprobs:
  * next-token top-k      -> POST /v1/completions with "logprobs" (widely supported)
  * teacher-forced scans  -> "prompt_logprobs" (vLLM extension, capability-probed)

Two call shapes matter:
  * prompt_logprobs=0    -> per prompt token, the logprob of the *realized* token
  * prompt_logprobs=k>0  -> per prompt token, the top-k distribution (the
    realized token is included in vLLM's dict, but not identifiable there, so
    realized-token logprobs always come from a separate k=0 call)

All clients share the same async interface so the suites are server-agnostic.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from .schemas import EndpointConfig


class ClientError(RuntimeError):
    pass


@dataclass
class CompletionResult:
    text: str
    # Generated positions: {"token": str, "logprob": float|None, "topk": {tok: lp}|None}
    token_logprobs: List[dict]
    # Prompt positions (index 0 is always None: no context). Entries are
    # {"logprob": float|None, "topk": {tok: lp}|None} depending on the request.
    prompt_scan: Optional[List[Optional[dict]]] = None


def make_client(cfg: EndpointConfig) -> "BaseClient":
    if cfg.backend == "mock":
        return MockClient(cfg)
    return OpenAICompatClient(cfg)


class BaseClient:
    supports_prompt_logprobs: Optional[bool] = None

    async def close(self) -> None:  # pragma: no cover - interface
        pass

    async def resolve_model(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    async def probe_prompt_logprobs(self) -> bool:
        if self.supports_prompt_logprobs is None:
            self.supports_prompt_logprobs = await self._probe()
        return self.supports_prompt_logprobs

    async def _probe(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_k: int = 0,
        prompt_logprobs: Optional[int] = None,
    ) -> CompletionResult:  # pragma: no cover - interface
        raise NotImplementedError

    async def next_token_topk(self, prompt: str, top_k: int = 20) -> Dict[str, float]:
        result = await self.complete(prompt, max_tokens=1, top_k=top_k)
        if not result.token_logprobs:
            return {}
        return result.token_logprobs[0]["topk"] or {}


class OpenAICompatClient(BaseClient):
    def __init__(self, cfg: EndpointConfig, read_timeout: float = 900.0):
        self.cfg = cfg
        self.model_id: Optional[str] = cfg.model
        self.supports_prompt_logprobs: Optional[bool] = None
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(read_timeout, connect=15.0),
            limits=httpx.Limits(max_connections=4),
        )

    async def close(self) -> None:
        await self._http.aclose()

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _url(self, path: str) -> str:
        return f"{self.cfg.base_url.rstrip('/')}{path}"

    async def _get(self, path: str) -> Any:
        try:
            resp = await self._http.get(self._url(path), headers=self._headers())
        except httpx.HTTPError as e:
            raise ClientError(f"{self.cfg.name}: request to {path} failed: {e}") from e
        if resp.status_code >= 400:
            raise ClientError(f"{self.cfg.name}: GET {path} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    async def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        try:
            resp = await self._http.post(self._url(path), headers=self._headers(), json=payload)
        except httpx.HTTPError as e:
            raise ClientError(f"{self.cfg.name}: request to {path} failed: {e}") from e
        if resp.status_code >= 400:
            raise ClientError(
                f"{self.cfg.name}: POST {path} -> {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()

    async def resolve_model(self) -> str:
        if not self.model_id:
            data = await self._get("/models")
            models = data.get("data") or []
            if not models:
                raise ClientError(f"{self.cfg.name}: no models returned by /v1/models")
            self.model_id = models[0]["id"]
        return self.model_id

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_k: int = 0,
        prompt_logprobs: Optional[int] = None,
    ) -> CompletionResult:
        payload: Dict[str, Any] = {
            "model": await self.resolve_model(),
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if top_k:
            payload["logprobs"] = top_k
        if prompt_logprobs is not None:
            payload["prompt_logprobs"] = prompt_logprobs
        try:
            data = await self._post("/completions", payload)
        except ClientError as e:
            # Some servers cap logprobs below what we asked for; retry small.
            if top_k > 5 and "logprob" in str(e).lower():
                return await self.complete(prompt, max_tokens, 5, prompt_logprobs)
            raise
        choice = (data.get("choices") or [{}])[0]

        lp = choice.get("logprobs") or {}
        tops = lp.get("top_logprobs") or []
        tok_list = lp.get("tokens") or []
        tok_lps = lp.get("token_logprobs") or []
        generated: List[dict] = []
        for i, tok in enumerate(tok_list):
            generated.append(
                {
                    "token": tok,
                    "logprob": tok_lps[i] if i < len(tok_lps) else None,
                    "topk": tops[i] if i < len(tops) else None,
                }
            )

        scan: Optional[List[Optional[dict]]] = None
        plp = choice.get("prompt_logprobs")
        if isinstance(plp, list):
            scan = []
            for element in plp:
                if not element:
                    scan.append(None)
                    continue
                # vLLM shape: {token_id: {"logprob": float, "rank": int, "decoded_token": str}}
                entries: Dict[str, float] = {}
                for key, val in element.items():
                    if isinstance(val, dict):
                        tok_str = val.get("decoded_token") or str(key)
                        entries[tok_str] = float(val["logprob"])
                    else:  # tolerate flat {token_str: logprob}
                        entries[str(key)] = float(val)
                if prompt_logprobs == 0:
                    # Only the realized token should be present.
                    first = next(iter(entries.values()), None)
                    scan.append({"logprob": first, "topk": None})
                else:
                    scan.append({"logprob": None, "topk": entries})
        return CompletionResult(
            text=choice.get("text") or "",
            token_logprobs=generated,
            prompt_scan=scan,
        )

    async def _probe(self) -> bool:
        try:
            r = await self.complete("hello", max_tokens=1, prompt_logprobs=0)
        except ClientError:
            return False
        return bool(r.prompt_scan) and len(r.prompt_scan) > 1


# ---------------------------------------------------------------------------
# Mock client: deterministic synthetic divergences for UI/demo/testing.
# Noise scales with KV dtype and with position, so comparisons behave
# qualitatively like real runs (bf16 ~ identical, fp4 ~ visibly divergent).
# ---------------------------------------------------------------------------

_MOCK_SIGMA = {
    "bf16": 0.0008,
    "auto": 0.0008,
    "fp8": 0.012,
    "fp8_e5m2": 0.030,
    "nvfp4": 0.090,
    "int4": 0.090,
    "mock": 0.020,
}


class MockClient(BaseClient):
    def __init__(self, cfg: EndpointConfig):
        self.cfg = cfg
        self.model_id: str = cfg.model or "mock-model-8b-instruct"
        self.supports_prompt_logprobs: bool = True
        self.sigma = _MOCK_SIGMA.get((cfg.kv_dtype or "").lower(), 0.02)

    async def resolve_model(self) -> str:
        return self.model_id

    async def _probe(self) -> bool:
        return True

    def _rng(self, *keys: Any) -> random.Random:
        blob = "|".join(str(k) for k in keys).encode("utf-8")
        return random.Random(int.from_bytes(hashlib.sha256(blob).digest()[:8], "big"))

    def _noise(self, position: int) -> float:
        return self.sigma * math.sqrt(1.0 + position / 512.0)

    def _clean_topk(self, prompt: str, position: int, k: int) -> Dict[str, float]:
        """Config-independent 'true' distribution, identical across endpoints."""
        rng = self._rng("clean", prompt, position)
        logits = [3.2 - 0.55 * i + rng.gauss(0, 1.0) for i in range(24)]
        mx = max(logits)
        exps = [math.exp(l - mx) for l in logits]
        total = sum(exps)
        lps = [math.log(e / total) for e in exps]
        toks = [f"tok{position % 5}_{i}" for i in range(24)]
        return dict(zip(toks[: max(1, k)], lps[: max(1, k)]))

    def _realize(self, clean: Dict[str, float], rng: random.Random, noise: float):
        tok = max(clean, key=clean.get)
        if clean.get(tok, 0.0) > -1e9 and rng.random() < min(0.5, noise * 3.0):
            ranked = sorted(clean, key=clean.get, reverse=True)
            if len(ranked) > 1:
                tok = ranked[1]
        return tok, clean[tok] + rng.gauss(0, noise)

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_k: int = 0,
        prompt_logprobs: Optional[int] = None,
    ) -> CompletionResult:
        await asyncio.sleep(0)
        plen = max(2, len(prompt.split()))
        scan: Optional[List[Optional[dict]]] = None
        if prompt_logprobs is not None:
            scan = [None]
            for pos in range(1, plen):
                clean = self._clean_topk(prompt, pos, 24)
                noise = self._noise(pos)
                rng = self._rng(self.cfg.name, prompt, pos)
                _tok, realized_lp = self._realize(clean, rng, noise)
                if prompt_logprobs == 0:
                    scan.append({"logprob": realized_lp, "topk": None})
                else:
                    k = min(prompt_logprobs, len(clean)) or len(clean)
                    topk = {t: lp + rng.gauss(0, noise * 0.5) for t, lp in list(clean.items())[:k]}
                    scan.append({"logprob": None, "topk": topk})

        generated: List[dict] = []
        parts: List[str] = []
        for step in range(max_tokens):
            pos = plen + step
            clean = self._clean_topk(prompt, pos, 24)
            noise = self._noise(pos)
            rng = self._rng(self.cfg.name, prompt, pos, "gen")
            tok, lp = self._realize(clean, rng, noise)
            k = top_k or 1
            topk = {t: lp2 + rng.gauss(0, noise * 0.5) for t, lp2 in list(clean.items())[:k]}
            topk[tok] = max(topk.get(tok, -1e9), lp)
            parts.append(tok)
            generated.append({"token": tok, "logprob": lp, "topk": topk})
        return CompletionResult(text=" ".join(parts), token_logprobs=generated, prompt_scan=scan)
