"""TCP client for the PYNQ-Z2 cosine skip-decision service."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

import numpy as np
import torch.nn.functional as F

from pynq_protocol import (
    CMD_CONFIG,
    CMD_PING,
    CMD_RESET,
    CMD_STEP,
    CONFIG,
    MAGIC,
    REQUEST,
    RESPONSE,
    STATUS_OK,
    phase_name,
)


FEATURE_DOWNSAMPLE_FACTOR = 2


@dataclass(frozen=True)
class PynqDecision:
    should_skip: bool = False
    cosine_passed: bool = False
    distance_passed: bool = False
    step_index: int = 0
    kernel_ms: float = 0.0
    server_ms: float = 0.0
    round_trip_ms: float = 0.0
    similarity: float = float("nan")
    normalized_distance: float = float("nan")
    skip_streak: int = 0
    threshold_q15: int = 0
    distance_threshold_q20: int = 0
    effective_threshold: float = 0.0
    effective_distance_threshold: float = 0.0
    threshold_requested: float = float("nan")
    threshold_phase: str = "none"
    adjacent_similarity_ema: float = float("nan")
    prepare_ms: float = 0.0
    feature_bytes: int = 0

    @property
    def threshold_passed(self) -> bool:
        """Compatibility alias for the original cosine-only result."""
        return self.cosine_passed


def compress_feature(
    feature, downsample_factor: int = FEATURE_DOWNSAMPLE_FACTOR
) -> np.ndarray:
    """Reduce one CHW latent feature and quantize it to symmetric int8."""
    if feature.ndim == 4 and feature.shape[0] in (1, 2):
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(f"Expected a CHW latent feature, got shape {feature.shape}")
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be positive")
    if (
        feature.shape[-2] % downsample_factor != 0
        or feature.shape[-1] % downsample_factor != 0
    ):
        raise ValueError(
            f"Feature size {feature.shape[-2:]} is not divisible by "
            f"downsample factor {downsample_factor}"
        )

    reduced = F.avg_pool2d(
        feature.detach().float().unsqueeze(0),
        kernel_size=downsample_factor,
        stride=downsample_factor,
    ).squeeze(0)
    values = reduced.cpu().numpy()
    values = np.ascontiguousarray(values.reshape(-1), dtype=np.float32)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if not np.isfinite(max_abs):
        raise ValueError("Latent feature contains NaN or infinity")
    if max_abs == 0.0:
        return np.zeros(values.shape, dtype=np.int8)
    scale = 127.0 / max_abs
    return np.ascontiguousarray(
        np.clip(np.rint(values * scale), -127, 127), dtype=np.int8
    )


def quantize_feature(feature) -> np.ndarray:
    """Backward-compatible alias for the int8 compression path."""
    return compress_feature(feature)


def encode_threshold(threshold: float) -> tuple[int, float]:
    """Convert a floating threshold to the FPGA Q1.15 representation."""
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("Cosine threshold must be finite")
    encoded = min(32767, max(0, int(threshold * 32768.0)))
    return encoded, encoded / 32768.0


def encode_distance_threshold(threshold: float) -> tuple[int, float]:
    """Convert a normalized-distance threshold to unsigned Q12.20."""
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("Distance threshold must be finite")
    encoded = min(0x7FFFFFFF, max(0, int(round(threshold * (1 << 20)))))
    return encoded, encoded / float(1 << 20)


class PynqCosineClient:
    """Persistent binary connection used once per diffusion timestep."""

    def __init__(self, host: str, port: int = 9000, timeout: float = 10.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self._socket: socket.socket | None = None
        self.total_bytes_sent = 0
        self.total_round_trip_ms = 0.0
        self.step_calls = 0

    def connect(self) -> None:
        if self._socket is not None:
            self.health_check()
            return
        sock = socket.create_connection((self.host, self.port), self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.timeout)
        self._socket = sock
        self._request(CMD_PING)

    def connect_with_retry(self, attempts: int = 5, delay_seconds: float = 2.0) -> int:
        """Connect and verify the protocol, returning the successful attempt."""
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        if delay_seconds < 0.0:
            raise ValueError("delay_seconds cannot be negative")

        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                self.connect()
                return attempt
            except (ConnectionError, OSError, TimeoutError) as exc:
                last_error = exc
                self.close()
                if attempt < attempts:
                    time.sleep(delay_seconds)
        raise ConnectionError(
            f"PYNQ health check failed after {attempts} attempts at "
            f"{self.host}:{self.port}: {last_error}"
        ) from last_error

    def health_check(self) -> float:
        """Verify an existing CSK4 connection and return round-trip time in ms."""
        return self._request(CMD_PING).round_trip_ms

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "PynqCosineClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def reset(self) -> None:
        self._request(CMD_RESET)
        self.total_bytes_sent = 0
        self.total_round_trip_ms = 0.0
        self.step_calls = 0

    def configure(
        self,
        *,
        enabled: bool,
        total_steps: int,
        warmup_steps: int,
        max_consecutive_skips: int,
        fixed_threshold: float,
        warmup_threshold: float,
        middle_threshold: float,
        late_start_ratio: float,
        late_margin: float,
        late_min: float,
        late_max: float,
        ema_alpha: float,
        distance_threshold: float,
    ) -> None:
        payload = CONFIG.pack(
            int(bool(enabled)),
            int(total_steps),
            int(warmup_steps),
            int(max_consecutive_skips),
            float(fixed_threshold),
            float(warmup_threshold),
            float(middle_threshold),
            float(late_start_ratio),
            float(late_margin),
            float(late_min),
            float(late_max),
            float(ema_alpha),
            float(distance_threshold),
        )
        self._request(CMD_CONFIG, length=len(payload), payload=payload)

    def decide(
        self,
        feature,
        step_index: int,
    ) -> PynqDecision:
        prepare_started = time.perf_counter()
        quantized = compress_feature(feature)
        prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
        response = self._request(
            CMD_STEP,
            step_index=int(step_index),
            length=int(quantized.size),
            payload=memoryview(quantized).cast("B"),
        )
        self.step_calls += 1
        self.total_bytes_sent += quantized.nbytes
        self.total_round_trip_ms += response.round_trip_ms
        return PynqDecision(
            should_skip=response.should_skip,
            cosine_passed=response.cosine_passed,
            distance_passed=response.distance_passed,
            step_index=response.step_index,
            kernel_ms=response.kernel_ms,
            server_ms=response.server_ms,
            round_trip_ms=response.round_trip_ms,
            similarity=response.similarity,
            normalized_distance=response.normalized_distance,
            skip_streak=response.skip_streak,
            threshold_q15=response.threshold_q15,
            distance_threshold_q20=response.distance_threshold_q20,
            effective_threshold=response.effective_threshold,
            effective_distance_threshold=response.effective_distance_threshold,
            threshold_requested=response.threshold_requested,
            threshold_phase=response.threshold_phase,
            adjacent_similarity_ema=response.adjacent_similarity_ema,
            prepare_ms=prepare_ms,
            feature_bytes=quantized.nbytes,
        )

    def _request(
        self,
        command: int,
        step_index: int = 0,
        length: int = 0,
        payload=None,
    ) -> PynqDecision:
        if self._socket is None:
            raise RuntimeError("PYNQ client is not connected")
        header = REQUEST.pack(
            MAGIC,
            command,
            step_index,
            length,
        )
        started = time.perf_counter()
        try:
            self._socket.sendall(header)
            if payload is not None:
                self._socket.sendall(payload)
            raw = self._recv_exact(RESPONSE.size)
        except (OSError, TimeoutError) as exc:
            self.close()
            raise ConnectionError(
                f"PYNQ communication failed at {self.host}:{self.port}: {exc}"
            ) from exc
        round_trip_ms = (time.perf_counter() - started) * 1000.0
        (
            magic,
            status,
            decision,
            cosine_passed,
            distance_passed,
            skip_streak,
            phase,
            reply_step,
            kernel_us,
            server_us,
            threshold_q15,
            distance_threshold_q20,
            similarity,
            normalized_distance,
            threshold_requested,
            adjacent_similarity_ema,
        ) = RESPONSE.unpack(raw)
        if magic != MAGIC:
            self.close()
            raise ConnectionError(f"Invalid PYNQ response magic: {magic!r}")
        if status != STATUS_OK:
            raise RuntimeError(f"PYNQ server rejected command {command}, status={status}")
        if command == CMD_STEP and reply_step != step_index:
            self.close()
            raise ConnectionError(
                f"PYNQ response step mismatch: expected {step_index}, got {reply_step}"
            )
        return PynqDecision(
            should_skip=bool(decision),
            cosine_passed=bool(cosine_passed),
            distance_passed=bool(distance_passed),
            step_index=reply_step,
            kernel_ms=kernel_us / 1000.0,
            server_ms=server_us / 1000.0,
            round_trip_ms=round_trip_ms,
            similarity=similarity,
            normalized_distance=normalized_distance,
            skip_streak=skip_streak,
            threshold_q15=threshold_q15,
            distance_threshold_q20=distance_threshold_q20,
            effective_threshold=threshold_q15 / 32768.0,
            effective_distance_threshold=distance_threshold_q20 / float(1 << 20),
            threshold_requested=threshold_requested,
            threshold_phase=phase_name(phase),
            adjacent_similarity_ema=adjacent_similarity_ema,
        )

    def _recv_exact(self, size: int) -> bytes:
        assert self._socket is not None
        data = bytearray(size)
        view = memoryview(data)
        received = 0
        while received < size:
            count = self._socket.recv_into(view[received:])
            if count == 0:
                raise ConnectionError("PYNQ server closed the connection")
            received += count
        return bytes(data)
