"""Real Groq streaming client.

- Streams SSE and timestamps: send, response headers, first token (any), first
  answer token (content), done. All via time.perf_counter().
- Reads rate-limit headers on every response and feeds a TokenBudget.
- On HTTP 429 writes the REAL body + headers to data/calibration/live_429_*.json
  (Authorization is never written).
- Retry policy: per-minute 429 -> wait retry-after if the deadline allows (max 2);
  per-day 429 -> never retry; 5xx / timeout -> 1 retry with jittered backoff.
  `retry_policy="naive"` disables the deadline/day checks so you can measure, on
  the real service, the case where retrying makes things worse.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from relay.live.ratelimit import HeaderSnapshot, TokenBudget, classify_429
from relay.live.sse import iter_sse_payloads, parse_payload

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
REASONING_MODELS_PREFIX = ("openai/gpt-oss",)

# on_delta(text) -> return False to abort the stream early (e.g. gate saw a violation)
DeltaCallback = Callable[[str], Awaitable[bool]]


class GroqCallError(Exception):
    def __init__(self, kind: str, status: int, message: str, retry_after_s: Optional[float] = None):
        super().__init__(f"{kind} ({status}): {message[:200]}")
        self.kind = kind  # rate_limit_minute | rate_limit_day | server | timeout | client | auth
        self.status = status
        self.message = message
        self.retry_after_s = retry_after_s


@dataclass
class CallResult:
    model: str
    text: str = ""
    reasoning_chars: int = 0
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    status_code: int = 0
    t_send: float = 0.0
    t_headers: Optional[float] = None
    t_first_any: Optional[float] = None
    t_first_content: Optional[float] = None
    t_done: Optional[float] = None
    retries: int = 0
    retry_wait_s: float = 0.0
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    error_kind: Optional[str] = None
    error_message: Optional[str] = None
    aborted: bool = False
    headers: Optional[Dict[str, Any]] = None
    request_id_groq: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error_kind is None

    def ms(self, t: Optional[float]) -> Optional[float]:
        return None if t is None else (t - self.t_send) * 1000.0


class GroqClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = GROQ_BASE_URL,
        connect_timeout_s: float = 5.0,
        stall_timeout_s: float = 8.0,
        capture_dir: Optional[Path] = Path("data/calibration"),
        retry_policy: str = "guarded",  # guarded | naive | none
        max_rate_limit_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set (export it or put it in .env; never commit it)")
        self.base_url = base_url.rstrip("/")
        self.connect_timeout_s = connect_timeout_s
        self.stall_timeout_s = stall_timeout_s
        self.capture_dir = capture_dir
        self.retry_policy = retry_policy
        self.max_rate_limit_retries = max_rate_limit_retries
        self._client = None
        self.captured_429: List[str] = []

    async def __aenter__(self) -> "GroqClient":
        import httpx  # lazy import keeps pure modules testable without httpx

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=self.connect_timeout_s, read=self.stall_timeout_s, write=10.0, pool=30.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()

    # ------------------------------------------------------------------ helpers
    def _capture_429(self, model: str, status: int, headers: Dict[str, str], body: str, now_wall: float) -> str:
        if not self.capture_dir:
            return ""
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        path = self.capture_dir / f"live_429_{int(now_wall * 1000)}.json"
        safe_headers = {k: v for k, v in headers.items() if k.lower() not in ("authorization", "set-cookie")}
        try:
            parsed_body: Any = json.loads(body)
        except Exception:
            parsed_body = body
        path.write_text(
            json.dumps(
                {
                    "is_live_capture": True,
                    "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_wall)),
                    "model": model,
                    "status_code": status,
                    "headers": safe_headers,
                    "body": parsed_body,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.captured_429.append(str(path))
        return str(path)

    def build_body(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": temperature,
            "stream": True,
        }
        if model.startswith(REASONING_MODELS_PREFIX):
            # Keep hidden reasoning short; it costs completion tokens and delays the first answer token.
            body["reasoning_effort"] = "low"
        if extra:
            body.update(extra)
        return body

    # ------------------------------------------------------------------ main call
    async def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        on_delta: Optional[DeltaCallback] = None,
        budget: Optional[TokenBudget] = None,
        deadline_at: Optional[float] = None,  # perf_counter() deadline
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> CallResult:
        import httpx

        assert self._client is not None, "use `async with GroqClient() as c:`"
        url = f"{self.base_url}/chat/completions"
        body = self.build_body(model, messages, max_tokens, temperature, extra_body)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        res = CallResult(model=model, t_send=time.perf_counter())
        server_retries = 0

        while True:
            attempt = {"t_start_ms": (time.perf_counter() - res.t_send) * 1000.0}
            try:
                async with self._client.stream("POST", url, json=body, headers=headers) as resp:
                    now = time.perf_counter()
                    res.t_headers = res.t_headers or now
                    res.status_code = resp.status_code
                    raw_headers = dict(resp.headers)
                    snap = HeaderSnapshot.from_headers(raw_headers, time.monotonic())
                    res.headers = snap.to_dict()
                    res.request_id_groq = raw_headers.get("x-request-id")
                    if budget is not None:
                        budget.observe(snap)
                    attempt.update(status=resp.status_code, headers=snap.to_dict())

                    if resp.status_code == 429:
                        text = (await resp.aread()).decode("utf-8", "replace")
                        try:
                            msg = json.loads(text).get("error", {}).get("message", text)
                        except Exception:
                            msg = text
                        kind, ra_msg = classify_429(msg)
                        retry_after = snap.retry_after_s or ra_msg
                        attempt.update(error="429", kind=kind, retry_after_s=retry_after,
                                       capture=self._capture_429(model, 429, raw_headers, text, time.time()))
                        res.attempts.append(attempt)
                        if budget is not None:
                            budget.observe_429(kind, retry_after)
                        if await self._should_retry_429(kind, retry_after, res, deadline_at):
                            continue
                        raise GroqCallError("rate_limit_day" if kind == "day" else "rate_limit_minute",
                                            429, msg, retry_after)

                    if resp.status_code in (401, 403):
                        text = (await resp.aread()).decode("utf-8", "replace")
                        raise GroqCallError("auth", resp.status_code, text)
                    if resp.status_code >= 500:
                        text = (await resp.aread()).decode("utf-8", "replace")
                        attempt.update(error=f"{resp.status_code}")
                        res.attempts.append(attempt)
                        if self.retry_policy != "none" and server_retries < 1 and self._deadline_allows(deadline_at, 1.5):
                            server_retries += 1
                            await self._backoff(res, 0.5 + random.random())
                            continue
                        raise GroqCallError("server", resp.status_code, text)
                    if resp.status_code >= 400:
                        text = (await resp.aread()).decode("utf-8", "replace")
                        raise GroqCallError("client", resp.status_code, text)

                    # ---- 200: consume stream ----
                    async for payload in iter_sse_payloads_async(resp.aiter_lines()):
                        delta = parse_payload(payload)
                        if delta is None:
                            break
                        t = time.perf_counter()
                        if delta.usage:
                            res.usage = delta.usage
                        if delta.finish_reason:
                            res.finish_reason = delta.finish_reason
                        if delta.reasoning:
                            res.reasoning_chars += len(delta.reasoning)
                            res.t_first_any = res.t_first_any or t
                        if delta.content:
                            res.t_first_any = res.t_first_any or t
                            res.t_first_content = res.t_first_content or t
                            res.text += delta.content
                            if on_delta is not None and (await on_delta(delta.content)) is False:
                                res.aborted = True
                                break
                        if deadline_at is not None and t > deadline_at:
                            res.aborted = True
                            res.error_kind = "timeout"
                            res.error_message = "deadline exceeded while streaming (slow success)"
                            break
                    res.t_done = time.perf_counter()
                    attempt["t_end_ms"] = (res.t_done - res.t_send) * 1000.0
                    res.attempts.append(attempt)
                    return res

            except GroqCallError as e:
                res.error_kind, res.error_message = e.kind, e.message
                res.t_done = time.perf_counter()
                return res
            except (httpx.TimeoutException, httpx.TransportError) as e:
                attempt.update(error=type(e).__name__)
                res.attempts.append(attempt)
                if res.t_first_content is not None:
                    # Stalled mid-stream: keep what we have, don't resend (would double-bill tokens).
                    res.error_kind, res.error_message = "timeout", f"stream stalled: {type(e).__name__}"
                    res.t_done = time.perf_counter()
                    return res
                if self.retry_policy != "none" and server_retries < 1 and self._deadline_allows(deadline_at, 2.0):
                    server_retries += 1
                    await self._backoff(res, 0.5 + random.random())
                    continue
                res.error_kind, res.error_message = "timeout", type(e).__name__
                res.t_done = time.perf_counter()
                return res

    def _deadline_allows(self, deadline_at: Optional[float], need_s: float) -> bool:
        if self.retry_policy == "naive" or deadline_at is None:
            return True
        return time.perf_counter() + need_s < deadline_at

    async def _should_retry_429(self, kind: str, retry_after: Optional[float], res: CallResult,
                                deadline_at: Optional[float]) -> bool:
        if self.retry_policy == "none" or res.retries >= self.max_rate_limit_retries:
            return False
        if self.retry_policy == "naive":
            # Naive: retry everything quickly, ignore retry-after and the day limit.
            await self._backoff(res, 0.2 + random.random() * 0.3)
            return True
        if kind == "day":
            return False  # retrying cannot succeed before reset
        wait = (retry_after if retry_after is not None else 1.0) + random.random() * 0.25
        if not self._deadline_allows(deadline_at, wait + 1.0):
            return False  # person would give up before the retry could answer
        await self._backoff(res, wait)
        return True

    async def _backoff(self, res: CallResult, seconds: float) -> None:
        res.retries += 1
        res.retry_wait_s += seconds
        await asyncio.sleep(seconds)


async def iter_sse_payloads_async(lines_aiter):  # small async adapter
    async for line in lines_aiter:
        for p in iter_sse_payloads([line]):
            yield p
