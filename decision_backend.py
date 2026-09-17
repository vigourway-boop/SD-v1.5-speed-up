"""Decision-backend selection for PYNQ hardware and the local PC fallback."""

from __future__ import annotations

from pc_skip_controller import PcCosineClient
from pynq_cosine_client import PynqCosineClient


class AutoDecisionClient:
    """Prefer PYNQ and switch to PC when the board cannot be reached."""

    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self._client = PynqCosineClient(host, port, timeout)
        self.fallback_used = False
        self.fallback_reason = None

    def __getattr__(self, name):
        return getattr(self._client, name)

    @property
    def backend_name(self):
        return self._client.backend_name

    @property
    def backend_display_name(self):
        if self.fallback_used and self.backend_name == "pc":
            return "PC CPU (automatic fallback)"
        return self._client.backend_display_name

    def connect_with_retry(self, attempts: int = 5, delay_seconds: float = 2.0) -> int:
        if self.backend_name == "pc":
            return self._client.connect_with_retry(attempts, delay_seconds)
        try:
            return self._client.connect_with_retry(attempts, delay_seconds)
        except (ConnectionError, OSError, TimeoutError) as exc:
            self._client.close()
            self.fallback_used = True
            self.fallback_reason = str(exc)
            fallback = PcCosineClient()
            fallback.fallback_used = True
            fallback.fallback_reason = self.fallback_reason
            self._client = fallback
            print(
                "PYNQ-Z2 is unavailable; switching to the PC CPU fallback. "
                "This run validates the skip algorithm, not FPGA performance."
            )
            return 1

    def close(self) -> None:
        self._client.close()


def create_decision_client(mode: str, host: str, port: int):
    mode = str(mode).lower()
    if mode == "pynq":
        return PynqCosineClient(host, port)
    if mode == "pc":
        return PcCosineClient()
    if mode == "auto":
        return AutoDecisionClient(host, port)
    raise ValueError(f"Unknown decision backend: {mode}")
