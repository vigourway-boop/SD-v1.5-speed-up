"""TCP client for the PYNQ-Z2 cosine skip-decision service."""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass

import numpy as np
import torch.nn.functional as F


MAGIC = b"CSK3"
CMD_RESET = 1
CMD_PING = 2
CMD_STEP = 3

REQUEST = struct.Struct("!4sB3xIIIII")
RESPONSE = struct.Struct("!4sBBBBIIId")
STATUS_OK = 0
FEATURE_DOWNSAMPLE_FACTOR = 2


@dataclass(frozen=True)
class PynqDecision:
    should_skip: bool
    threshold_passed: bool
    step_index: int
    kernel_ms: float
    server_ms: float
    round_trip_ms: float
    similarity: float
    skip_streak: int
    prepare_ms: float = 0.0
    feature_bytes: int = 0


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
    """Backward-compatible alias for the CSK3 compression path."""
    return compress_feature(feature)


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
            return
        sock = socket.create_connection((self.host, self.port), self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.timeout)
        self._socket = sock
        self._request(CMD_PING)

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

    def decide(
        self,
        feature,
        step_index: int,
        threshold: float,
        warmup_steps: int,
        max_consecutive_skips: int,
    ) -> PynqDecision:
        prepare_started = time.perf_counter()
        quantized = compress_feature(feature)
        prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
        response = self._request(
            CMD_STEP,
            step_index=int(step_index),
            length=int(quantized.size),
            threshold_q15=min(32767, max(0, int(float(threshold) * 32768.0))),
            warmup_steps=int(warmup_steps),
            max_consecutive_skips=int(max_consecutive_skips),
            payload=memoryview(quantized).cast("B"),
        )
        self.step_calls += 1
        self.total_bytes_sent += quantized.nbytes
        self.total_round_trip_ms += response.round_trip_ms
        return PynqDecision(
            should_skip=response.should_skip,
            threshold_passed=response.threshold_passed,
            step_index=response.step_index,
            kernel_ms=response.kernel_ms,
            server_ms=response.server_ms,
            round_trip_ms=response.round_trip_ms,
            similarity=response.similarity,
            skip_streak=response.skip_streak,
            prepare_ms=prepare_ms,
            feature_bytes=quantized.nbytes,
        )

    def _request(
        self,
        command: int,
        step_index: int = 0,
        length: int = 0,
        threshold_q15: int = 0,
        warmup_steps: int = 0,
        max_consecutive_skips: int = 0,
        payload=None,
    ) -> PynqDecision:
        if self._socket is None:
            raise RuntimeError("PYNQ client is not connected")
        header = REQUEST.pack(
            MAGIC,
            command,
            step_index,
            length,
            threshold_q15,
            warmup_steps,
            max_consecutive_skips,
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
            threshold_passed,
            skip_streak,
            reply_step,
            kernel_us,
            server_us,
            similarity,
        ) = RESPONSE.unpack(raw)
        if magic != MAGIC:
            raise RuntimeError(f"Invalid PYNQ response magic: {magic!r}")
        if status != STATUS_OK:
            raise RuntimeError(f"PYNQ server rejected command {command}, status={status}")
        if command == CMD_STEP and reply_step != step_index:
            raise RuntimeError(
                f"PYNQ response step mismatch: expected {step_index}, got {reply_step}"
            )
        return PynqDecision(
            should_skip=bool(decision),
            threshold_passed=bool(threshold_passed),
            step_index=reply_step,
            kernel_ms=kernel_us / 1000.0,
            server_ms=server_us / 1000.0,
            round_trip_ms=round_trip_ms,
            similarity=similarity,
            skip_streak=skip_streak,
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
