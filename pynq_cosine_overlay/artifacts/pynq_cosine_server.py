#!/usr/bin/env python3
"""PYNQ-Z2 CSK4 service for board-controlled diffusion skip decisions."""

import argparse
import math
import socket
import time

import numpy as np
from pynq import Overlay, allocate

from dynamic_threshold import DynamicThresholdController
from pynq_protocol import (
    CMD_CONFIG,
    CMD_PING,
    CMD_RESET,
    CMD_STEP,
    CONFIG,
    MAGIC,
    PHASE_NONE,
    REQUEST,
    RESPONSE,
    STATUS_CONFIG_ERROR,
    STATUS_HARDWARE_ERROR,
    STATUS_LENGTH_ERROR,
    STATUS_NOT_CONFIGURED,
    STATUS_OK,
    STATUS_PROTOCOL_ERROR,
    phase_code,
)


CTRL = 0x00
AP_RETURN = 0x10
X_LOW = 0x18
LENGTH = 0x24
STEP_INDEX = 0x2C
THRESHOLD_Q15 = 0x34
WARMUP_STEPS = 0x3C
MAX_CONSECUTIVE_SKIPS = 0x44
RESET_STATE = 0x4C
DOT_OUT_LOW = 0x54
NORM_X_LOW = 0x6C
NORM_Y_LOW = 0x84
COSINE_PASSED_OUT = 0x9C
SKIP_STREAK_OUT = 0xAC
# Verified against the generated CSK4 HWH after synthesis.
DISTANCE_THRESHOLD_Q20 = 0xBC
DISTANCE_PASSED_OUT = 0xC4

MAX_VECTOR_LEN = 4096
IP_TIMEOUT_SECONDS = 1.0


class HardwareTimeout(RuntimeError):
    pass


def recv_exact(connection, target):
    view = target if isinstance(target, memoryview) else memoryview(target)
    if view.format != "B":
        view = view.cast("B")
    received = 0
    while received < len(view):
        count = connection.recv_into(view[received:])
        if count == 0:
            raise ConnectionError("client disconnected")
        received += count


def encode_cosine_threshold(value):
    if not math.isfinite(value):
        raise ValueError("cosine threshold must be finite")
    return min(32767, max(0, int(value * 32768.0)))


def encode_distance_threshold(value):
    if not math.isfinite(value) or not 0.0 <= value <= 2.0:
        raise ValueError("distance threshold must be finite and within [0, 2]")
    return int(round(value * (1 << 20)))


class CosineHardware:
    def __init__(self, bitstream):
        self.overlay = Overlay(bitstream)
        self.ip = self.overlay.cosine_skip_0
        self.incoming = allocate(shape=(MAX_VECTOR_LEN,), dtype=np.int8)
        self.reset()

    def close(self):
        self.incoming.freebuffer()

    def reset(self):
        self._run_ip(
            self.incoming,
            length=0,
            step_index=0,
            threshold_q15=0,
            warmup_steps=0,
            max_skips=0,
            reset_state=1,
            distance_threshold_q20=0,
        )

    def receive_feature(self, connection, length):
        target = memoryview(self.incoming).cast("B")[:length]
        recv_exact(connection, target)
        self.incoming.flush()

    def decide(
        self,
        step_index,
        length,
        threshold_q15,
        warmup_steps,
        max_skips,
        distance_threshold_q20,
    ):
        started = time.perf_counter_ns()
        kernel_started = time.perf_counter_ns()
        (
            should_skip,
            cosine_passed,
            distance_passed,
            skip_streak,
            dot,
            norm_x,
            norm_y,
        ) = self._run_ip(
            self.incoming,
            length,
            step_index,
            threshold_q15,
            warmup_steps,
            max_skips,
            reset_state=0,
            distance_threshold_q20=distance_threshold_q20,
        )
        kernel_us = (time.perf_counter_ns() - kernel_started) // 1000
        similarity = math.nan
        normalized_distance = math.nan
        if norm_x and norm_y:
            similarity = dot / math.sqrt(norm_x * norm_y)
            similarity = max(-1.0, min(1.0, similarity))
            norm_sum = norm_x + norm_y
            distance_num = max(0, norm_sum - 2 * dot)
            normalized_distance = distance_num / float(norm_sum)
        server_us = (time.perf_counter_ns() - started) // 1000
        return (
            bool(should_skip),
            bool(cosine_passed),
            bool(distance_passed),
            int(skip_streak),
            similarity,
            normalized_distance,
            kernel_us,
            server_us,
        )

    def _write_u64(self, offset_low, value):
        value = int(value)
        self.ip.write(offset_low, value & 0xFFFFFFFF)
        self.ip.write(offset_low + 4, (value >> 32) & 0xFFFFFFFF)

    def _read_u64(self, offset_low):
        return self.ip.read(offset_low) | (self.ip.read(offset_low + 4) << 32)

    def _read_i64(self, offset_low):
        value = self._read_u64(offset_low)
        return value - (1 << 64) if value & (1 << 63) else value

    def _run_ip(
        self,
        x_buffer,
        length,
        step_index,
        threshold_q15,
        warmup_steps,
        max_skips,
        reset_state,
        distance_threshold_q20,
    ):
        self._write_u64(X_LOW, x_buffer.physical_address)
        self.ip.write(LENGTH, length)
        self.ip.write(STEP_INDEX, step_index)
        self.ip.write(THRESHOLD_Q15, threshold_q15)
        self.ip.write(WARMUP_STEPS, warmup_steps)
        self.ip.write(MAX_CONSECUTIVE_SKIPS, max_skips)
        self.ip.write(RESET_STATE, reset_state)
        self.ip.write(DISTANCE_THRESHOLD_Q20, distance_threshold_q20)
        self.ip.write(CTRL, 0x01)
        deadline = time.monotonic() + IP_TIMEOUT_SECONDS
        while (self.ip.read(CTRL) & 0x2) == 0:
            if time.monotonic() >= deadline:
                raise HardwareTimeout("cosine_skip IP did not finish within 1 second")
        return (
            self.ip.read(AP_RETURN),
            self.ip.read(COSINE_PASSED_OUT),
            self.ip.read(DISTANCE_PASSED_OUT),
            self.ip.read(SKIP_STREAK_OUT),
            self._read_i64(DOT_OUT_LOW),
            self._read_u64(NORM_X_LOW),
            self._read_u64(NORM_Y_LOW),
        )


class BoardController:
    def __init__(self):
        self.threshold = None
        self.warmup_steps = 0
        self.max_skips = 0
        self.distance_threshold_q20 = 0

    def configure(self, values):
        (
            enabled,
            total_steps,
            warmup_steps,
            max_skips,
            fixed_threshold,
            warmup_threshold,
            middle_threshold,
            late_start_ratio,
            late_margin,
            late_min,
            late_max,
            ema_alpha,
            distance_threshold,
        ) = values
        if max_skips < 0:
            raise ValueError("max_consecutive_skips cannot be negative")
        self.threshold = DynamicThresholdController(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            enabled=bool(enabled),
            fixed_threshold=fixed_threshold,
            warmup_threshold=warmup_threshold,
            middle_threshold=middle_threshold,
            late_start_ratio=late_start_ratio,
            late_margin=late_margin,
            late_min=late_min,
            late_max=late_max,
            ema_alpha=ema_alpha,
        )
        self.warmup_steps = int(warmup_steps)
        self.max_skips = int(max_skips)
        self.distance_threshold_q20 = encode_distance_threshold(distance_threshold)

    def reset(self):
        if self.threshold is not None:
            self.threshold.reset()


def send_response(
    connection,
    status,
    decision=0,
    cosine_passed=0,
    distance_passed=0,
    skip_streak=0,
    phase=PHASE_NONE,
    step=0,
    kernel_us=0,
    server_us=0,
    threshold_q15=0,
    distance_threshold_q20=0,
    similarity=math.nan,
    normalized_distance=math.nan,
    threshold_requested=math.nan,
    adjacent_similarity_ema=math.nan,
):
    connection.sendall(
        RESPONSE.pack(
            MAGIC,
            status,
            int(bool(decision)),
            int(bool(cosine_passed)),
            int(bool(distance_passed)),
            min(255, max(0, int(skip_streak))),
            int(phase),
            int(step),
            int(kernel_us),
            int(server_us),
            int(threshold_q15),
            int(distance_threshold_q20),
            float(similarity),
            float(normalized_distance),
            float(threshold_requested),
            float(adjacent_similarity_ema),
        )
    )


def serve_client(connection, address, hardware):
    print("Client connected:", address, flush=True)
    controller = BoardController()
    hardware.reset()
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.settimeout(15.0)
    while True:
        header = bytearray(REQUEST.size)
        recv_exact(connection, header)
        magic, command, step, length = REQUEST.unpack(header)
        if magic != MAGIC:
            send_response(connection, STATUS_PROTOCOL_ERROR, step=step)
            return
        if command == CMD_PING:
            send_response(connection, STATUS_OK, step=step)
            continue
        if command == CMD_RESET:
            hardware.reset()
            controller.reset()
            send_response(connection, STATUS_OK, step=step)
            continue
        if command == CMD_CONFIG:
            if length != CONFIG.size:
                send_response(connection, STATUS_LENGTH_ERROR, step=step)
                return
            raw_config = bytearray(CONFIG.size)
            recv_exact(connection, raw_config)
            try:
                controller.configure(CONFIG.unpack(raw_config))
                hardware.reset()
            except ValueError as exc:
                print("Invalid controller configuration:", exc, flush=True)
                send_response(connection, STATUS_CONFIG_ERROR, step=step)
                continue
            send_response(connection, STATUS_OK, step=step)
            continue
        if command != CMD_STEP:
            send_response(connection, STATUS_PROTOCOL_ERROR, step=step)
            return
        if controller.threshold is None:
            send_response(connection, STATUS_NOT_CONFIGURED, step=step)
            return
        if length <= 0 or length > MAX_VECTOR_LEN:
            send_response(connection, STATUS_LENGTH_ERROR, step=step)
            return

        hardware.receive_feature(connection, length)
        try:
            point = controller.threshold.point_for(step)
            threshold_q15 = encode_cosine_threshold(point.value)
            (
                decision,
                cosine_passed,
                distance_passed,
                skip_streak,
                similarity,
                normalized_distance,
                kernel_us,
                server_us,
            ) = hardware.decide(
                step,
                length,
                threshold_q15,
                controller.warmup_steps,
                controller.max_skips,
                controller.distance_threshold_q20,
            )
            controller.threshold.observe(similarity, decision)
        except HardwareTimeout as exc:
            print("FPGA timeout:", exc, flush=True)
            send_response(connection, STATUS_HARDWARE_ERROR, step=step)
            raise
        except ValueError as exc:
            print("Invalid step:", exc, flush=True)
            send_response(connection, STATUS_CONFIG_ERROR, step=step)
            continue

        adjacent_ema = point.adjacent_similarity_ema
        send_response(
            connection,
            STATUS_OK,
            decision,
            cosine_passed,
            distance_passed,
            skip_streak,
            phase_code(point.phase),
            step,
            kernel_us,
            server_us,
            threshold_q15,
            controller.distance_threshold_q20,
            similarity,
            normalized_distance,
            point.value,
            math.nan if adjacent_ema is None else adjacent_ema,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--bitstream", default="cosine_overlay.bit")
    args = parser.parse_args()

    hardware = CosineHardware(args.bitstream)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print("PYNQ CSK4 server listening on {}:{}".format(args.host, args.port), flush=True)
    try:
        while True:
            connection, address = server.accept()
            try:
                with connection:
                    serve_client(connection, address, hardware)
            except (ConnectionError, OSError) as exc:
                print("Client disconnected:", exc, flush=True)
            finally:
                hardware.reset()
    finally:
        server.close()
        hardware.close()


if __name__ == "__main__":
    main()
