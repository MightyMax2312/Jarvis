"""Kev router: fast closed-set action classification via /v1/systemone.

kev-0.5b is a prefill-only pointer model -- it cannot emit text, so it can only
pick one of a declared option set. Measured 9/10 on the desktop verb set at
~53 ms warm on CUDA, with correct answers at confidence >= 0.77 and its single
miss at 0.49. That separation is what makes the threshold below meaningful.
"""

import time
from dataclasses import dataclass, field

import aiohttp
from loguru import logger

from .vocab import ACTION_CHOICES, ACTION_KEYWORDS, SELF_CONTAINED_ROUTER

JEV_ENDPOINT = "http://127.0.0.1:8000/v1/systemone"


@dataclass
class RouterVerdict:
    action: str | None = None
    confidence: float = 0.0
    latency: float = 0.0
    reason: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        return bool(self.action) and self.action != "NONE"


class KevRouter:
    def __init__(
        self,
        endpoint: str = JEV_ENDPOINT,
        threshold: float = 0.70,
        timeout: float = 15.0,
        session: aiohttp.ClientSession | None = None,
    ):
        self._endpoint = endpoint
        self._threshold = threshold
        self._timeout = timeout
        self._session = session
        self._owned = session is None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
            self._owned = True
        return self._session

    async def close(self) -> None:
        if self._owned and self._session and not self._session.closed:
            await self._session.close()

    @staticmethod
    def worth_routing(user_text: str) -> bool:
        """Cheap pre-filter so pure chat never pays the router call."""
        low = user_text.lower()
        return any(kw in low for kw in ACTION_KEYWORDS)

    async def route(self, user_text: str, context: str = "") -> RouterVerdict:
        if not self.worth_routing(user_text):
            return RouterVerdict(reason="prefilter")

        payload = {
            "state": f"User intent: {user_text}. Context: {context or 'unknown'}",
            "questions": {"action": {"type": "choice", "criteria": ACTION_CHOICES}},
        }
        session = await self._ensure_session()
        t0 = time.perf_counter()
        try:
            async with session.post(self._endpoint, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as e:
            logger.warning(f"[router] unavailable: {e}")
            return RouterVerdict(reason=f"error: {e}")

        latency = time.perf_counter() - t0
        answer = data.get("answers", {}).get("action", {})
        verdict = RouterVerdict(
            action=answer.get("choice"),
            confidence=float(answer.get("confidence") or 0.0),
            latency=latency,
            probabilities=answer.get("probabilities") or {},
        )

        if not verdict.actionable:
            verdict.reason = "router said NONE"
        elif verdict.action not in SELF_CONTAINED_ROUTER:
            verdict.reason = f"{verdict.action} needs arguments, defer to LLM"
        elif verdict.confidence < self._threshold:
            verdict.reason = (
                f"conf {verdict.confidence:.3f} < {self._threshold:.2f}, defer to LLM"
            )
        else:
            verdict.reason = "fast path"

        logger.info(
            "[router] {:.0f}ms {} conf={:.3f} -> {}",
            latency * 1000,
            "OK" if verdict.reason == "fast path" else "defer",
            verdict.confidence,
            verdict.action,
        )
        return verdict
