"""CPU implementation of the CSK4 board controller and FPGA decision logic."""

from __future__ import annotations

import math
import struct
import time

import numpy as np

from dynamic_threshold import DynamicThresholdController
from pynq_cosine_client import (
    PynqDecision,
    compress_feature,
    encode_distance_threshold,
    encode_threshold,
)


class PcCosineClient:
    """Drop-in local replacement for ``PynqCosineClient``.

    The integer comparisons and state updates mirror ``cosine_skip.cpp`` so
    algorithm development can continue without a connected PYNQ-Z2.
    """

    backend_name = "pc"
    backend_display_name = "PC CPU"
    threshold_controller_location = "PC CPU"
    cosine_location = "PC CPU"
    distance_location = "PC CPU"
    skip_controller_location = "PC CPU"
    fallback_used = False
    fallback_reason = None

    def __init__(self):
        self._controller = None
        self._reference = None
        self._warmup_steps = 0
        self._max_consecutive_skips = 0
        self._skip_streak = 0
        self._distance_threshold_q20 = 0
        self.total_bytes_sent = 0
        self.total_round_trip_ms = 0.0
        self.step_calls = 0

    def connect(self) -> None:
        return None

    def connect_with_retry(self, attempts: int = 1, delay_seconds: float = 0.0) -> int:
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        if delay_seconds < 0.0:
            raise ValueError("delay_seconds cannot be negative")
        return 1

    def health_check(self) -> float:
        return 0.0

    def close(self) -> None:
        return None

    def reset(self) -> None:
        self._reference = None
        self._skip_streak = 0
        if self._controller is not None:
            self._controller.reset()
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
        if max_consecutive_skips < 0:
            raise ValueError("max_consecutive_skips cannot be negative")
        if not math.isfinite(distance_threshold) or not 0.0 <= distance_threshold <= 2.0:
            raise ValueError("distance_threshold must be finite and within [0, 2]")
        # CONFIG transmits float32 values; use those same values on the PC.
        (fixed_threshold, warmup_threshold, middle_threshold, late_start_ratio,
         late_margin, late_min, late_max, ema_alpha, distance_threshold) = struct.unpack(
            "!9f", struct.pack("!9f", fixed_threshold, warmup_threshold,
                               middle_threshold, late_start_ratio, late_margin,
                               late_min, late_max, ema_alpha, distance_threshold)
        )
        self._controller = DynamicThresholdController(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            enabled=enabled,
            fixed_threshold=fixed_threshold,
            warmup_threshold=warmup_threshold,
            middle_threshold=middle_threshold,
            late_start_ratio=late_start_ratio,
            late_margin=late_margin,
            late_min=late_min,
            late_max=late_max,
            ema_alpha=ema_alpha,
        )
        self._warmup_steps = int(warmup_steps)
        self._max_consecutive_skips = int(max_consecutive_skips)
        self._distance_threshold_q20 = encode_distance_threshold(distance_threshold)[0]
        self.reset()

    def decide(self, feature, step_index: int) -> PynqDecision:
        if self._controller is None:
            raise RuntimeError("PC skip controller is not configured")

        prepare_started = time.perf_counter()
        quantized = compress_feature(feature)
        prepare_ms = (time.perf_counter() - prepare_started) * 1000.0

        decision_started = time.perf_counter()
        point = self._controller.point_for(int(step_index))
        threshold_q15, effective_threshold = encode_threshold(point.value)
        comparable = (
            self._reference is not None
            and self._reference.size == quantized.size
        )
        dot = 0
        norm_x = 0
        norm_y = 0
        similarity = math.nan
        normalized_distance = math.nan
        cosine_passed = False
        distance_passed = False

        if comparable:
            current = quantized.astype(np.int64, copy=False)
            reference = self._reference.astype(np.int64, copy=False)
            dot = int(np.dot(current, reference))
            norm_x = int(np.dot(current, current))
            norm_y = int(np.dot(reference, reference))

            if norm_x and norm_y:
                similarity = dot / math.sqrt(norm_x * norm_y)
                similarity = max(-1.0, min(1.0, similarity))
                norm_sum = norm_x + norm_y
                distance_num = max(0, norm_sum - 2 * dot)
                normalized_distance = distance_num / float(norm_sum)

                if dot > 0:
                    left = dot * dot
                    right = (norm_x * norm_y * threshold_q15 * threshold_q15) >> 30
                    cosine_passed = left > right
            if norm_x or norm_y:
                norm_sum = norm_x + norm_y
                distance_num = max(0, norm_sum - 2 * dot)
                distance_passed = (
                    (distance_num << 20)
                    <= self._distance_threshold_q20 * norm_sum
                )

        should_skip = (
            comparable
            and cosine_passed
            and distance_passed
            and int(step_index) >= self._warmup_steps
            and self._skip_streak < self._max_consecutive_skips
        )
        if should_skip:
            self._skip_streak += 1
        else:
            self._reference = quantized.copy()
            self._skip_streak = 0

        self._controller.observe(similarity, should_skip)
        decision_ms = (time.perf_counter() - decision_started) * 1000.0
        self.step_calls += 1
        self.total_bytes_sent += quantized.nbytes

        return PynqDecision(
            should_skip=should_skip,
            cosine_passed=cosine_passed,
            distance_passed=distance_passed,
            step_index=int(step_index),
            similarity=similarity,
            normalized_distance=normalized_distance,
            skip_streak=self._skip_streak,
            threshold_q15=threshold_q15,
            distance_threshold_q20=self._distance_threshold_q20,
            effective_threshold=effective_threshold,
            effective_distance_threshold=(
                self._distance_threshold_q20 / float(1 << 20)
            ),
            threshold_requested=point.value,
            threshold_phase=point.phase,
            adjacent_similarity_ema=(
                math.nan
                if point.adjacent_similarity_ema is None
                else point.adjacent_similarity_ema
            ),
            prepare_ms=prepare_ms,
            feature_bytes=quantized.nbytes,
            decision_ms=decision_ms,
        )
