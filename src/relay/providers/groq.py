"""Live Groq API client adapter with async SSE streaming and header tracking.

Implements ModelProviderPort for Groq's OpenAI-compatible completions endpoint.
Captures monotonic TTFT arrival on the first content token, extracts usage metadata,
and updates live rate-limit quota snapshots.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx
from pydantic import SecretStr

from relay.contracts.errors import (
    DailyBudgetExhaustedError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from relay.observability.logger import sanitize_message
from relay.providers.base import (
    ModelProviderPort,
    ProviderCompletionRequest,
    ProviderQuotaSnapshot,
    ProviderStreamChunk,
)
from relay.providers.headers import parse_groq_429_payload, update_snapshot_from_headers

logger = logging.getLogger("relay.providers.groq")


class GroqProviderAdapter:
    """Async client adapter for Groq inference on the free tier."""

    def __init__(
        self,
        api_key: Optional[SecretStr] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        default_model: str = "llama-3.1-8b-instant",
        request_timeout_s: float = 15.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.request_timeout_s = request_timeout_s
        self._external_client = http_client
        self._quota_snapshot = ProviderQuotaSnapshot()

    def get_quota_snapshot(self) -> ProviderQuotaSnapshot:
        """Return the latest tracked rate-limit quota snapshot."""
        return self._quota_snapshot

    def _get_api_key_str(self) -> Optional[str]:
        """Unmask API key string if available."""
        if self.api_key:
            return self.api_key.get_secret_value()
        return None

    async def check_health(self) -> bool:
        """Lightweight connectivity probe verifying Groq API availability."""
        key_str = self._get_api_key_str()
        if not key_str:
            return False

        headers = {
            "Authorization": f"Bearer {key_str}",
            "User-Agent": "TheRelay/0.1.0",
        }
        url = f"{self.base_url}/models"

        try:
            client = self._external_client or httpx.AsyncClient(timeout=self.request_timeout_s)
            try:
                resp = await client.get(url, headers=headers)
                if resp.headers:
                    self._quota_snapshot = update_snapshot_from_headers(
                        self._quota_snapshot, resp.headers
                    )
                return resp.status_code == 200
            finally:
                if self._external_client is None:
                    await client.aclose()
        except Exception as e:
            logger.warning("Groq health check failed: %s", sanitize_message(str(e)))
            return False

    async def stream_completion(
        self, request: ProviderCompletionRequest
    ) -> AsyncIterator[ProviderStreamChunk]:
        """Stream generated completion chunks with precise first-token timing.

        Yields ProviderStreamChunk objects containing text delta, first-token flag,
        monotonic arrival timestamp, and final token usage counters.
        """
        key_str = self._get_api_key_str()
        if not key_str:
            raise ProviderAuthenticationError(
                "Cannot initiate Groq inference: GROQ_API_KEY is not configured"
            )

        headers = {
            "Authorization": f"Bearer {key_str}",
            "Content-Type": "application/json",
            "User-Agent": "TheRelay/0.1.0",
        }

        payload: Dict[str, Any] = {
            "model": request.model_id or self.default_model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        url = f"{self.base_url}/chat/completions"
        timeout = httpx.Timeout(request.timeout_s or self.request_timeout_s)

        client = self._external_client or httpx.AsyncClient(timeout=timeout)
        resp: Optional[httpx.Response] = None
        try:
            req = client.build_request("POST", url, headers=headers, json=payload)
            resp = await client.send(req, stream=True)

            try:
                # Update tracked rate-limit telemetry from headers
                if resp.headers:
                    self._quota_snapshot = update_snapshot_from_headers(
                        self._quota_snapshot, resp.headers
                    )

                # Handle non-200 HTTP responses
                if resp.status_code != 200:
                    await resp.aread()
                    status = resp.status_code

                    # If model_not_found (e.g. account catalog uses different model ID), fallback to active chat model
                    if status == 404 and "model_not_found" in resp.text and payload["model"] != "qwen/qwen3.8-27b":
                        fallback_model = "qwen/qwen3.8-27b"
                        logger.warning(
                            "Model '%s' not found on Groq (HTTP 404 model_not_found); seamlessly falling back to available chat model '%s'",
                            payload["model"],
                            fallback_model,
                        )
                        payload["model"] = fallback_model
                        req2 = client.build_request("POST", url, headers=headers, json=payload)
                        resp2 = await client.send(req2, stream=True)
                        if resp2.status_code == 200:
                            resp = resp2
                            status = 200
                            if resp.headers:
                                self._quota_snapshot = update_snapshot_from_headers(
                                    self._quota_snapshot, resp.headers
                                )
                        else:
                            await resp2.aread()
                            status = resp2.status_code
                            resp = resp2

                    if status != 200:
                        if status == 401:
                            raise ProviderAuthenticationError(
                                "Groq rejected API key: HTTP 401 Unauthorized"
                            )

                        if status == 429:
                            error_json: Any = None
                            try:
                                error_json = resp.json()
                            except Exception:
                                error_json = resp.text

                            parsed_429 = parse_groq_429_payload(error_json)
                            retry_s = parsed_429.get("retry_after_s")

                            if parsed_429.get("is_daily"):
                                raise DailyBudgetExhaustedError(
                                    message=f"Groq daily quota exhausted: {parsed_429.get('message')}"
                                )
                            raise RateLimitError(
                                message=f"Groq rate limit exceeded: {parsed_429.get('message')}",
                                retry_after_s=retry_s,
                                limit_type=parsed_429.get("metric"),
                                is_daily=False,
                            )

                        if status in (500, 502, 503, 504):
                            raise ProviderUnavailableError(
                                message=f"Groq service unavailable: HTTP {status}",
                                status_code=status,
                            )

                        raise ProviderError(
                            message=f"Groq API call failed: HTTP {status} {resp.text}",
                            status_code=status,
                        )

                # Stream Server-Sent Events (SSE)
                token_index = 0
                is_first = True
                prompt_tokens: Optional[int] = None
                completion_tokens: Optional[int] = None

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    raw_data = line[5:].strip()
                    if raw_data == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue

                    # Check usage metadata chunk
                    if "usage" in chunk_data and chunk_data["usage"]:
                        usage = chunk_data["usage"]
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")

                    # Extract choices delta
                    choices = chunk_data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    text_content = delta.get("content", "")
                    finish_reason = choices[0].get("finish_reason")

                    if text_content:
                        arrival_t = time.perf_counter()
                        token_index += 1
                        yield ProviderStreamChunk(
                            text_delta=text_content,
                            is_first_token=is_first,
                            token_index=token_index,
                            arrival_monotonic_s=arrival_t,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            finish_reason=finish_reason,
                        )
                        if is_first:
                            is_first = False

                # If finish_reason was emitted on a final chunk without text, emit final chunk with usage
                if prompt_tokens is not None or completion_tokens is not None:
                    yield ProviderStreamChunk(
                        text_delta="",
                        is_first_token=False,
                        token_index=token_index,
                        arrival_monotonic_s=time.perf_counter(),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        finish_reason="stop",
                    )
            finally:
                if resp is not None:
                    await resp.aclose()

        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout) as e:
            raise ProviderTimeoutError(
                message=f"Groq streaming connection timed out: {e}"
            ) from e
        except httpx.NetworkError as e:
            raise ProviderUnavailableError(
                message=f"Groq network connection error: {e}"
            ) from e
        finally:
            if self._external_client is None:
                await client.aclose()
