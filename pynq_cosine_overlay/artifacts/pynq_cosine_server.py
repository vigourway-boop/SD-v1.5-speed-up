#!/usr/bin/env python3
"""PYNQ-Z2 TCP service for hardware cosine skip decisions."""

import argparse
import math
import socket
import struct
import time

import numpy as np
from pynq import Overlay, allocate


MAGIC = b"CSK3"
CMD_RESET = 1
CMD_PING = 2
CMD_STEP = 3
REQUEST = struct.Struct("!4sB3xIIIII")
RESPONSE = struct.Struct("!4sBBBBIIId")
STATUS_OK = 0
STATUS_PROTOCOL_ERROR = 1
STATUS_LENGTH_ERROR = 2
STATUS_HARDWARE_ERROR = 3

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
THRESHOLD_PASSED_OUT = 0x9C
SKIP_STREAK_OUT = 0xAC
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
        )

    def receive_feature(self, connection, length):
        target = memoryview(self.incoming).cast("B")[:length]
        recv_exact(connection, target)
        self.incoming.flush()

    def decide(self, step_index, length, threshold_q15, warmup_steps, max_skips):
        started = time.perf_counter_ns()
        kernel_started = time.perf_counter_ns()
        (
            should_skip,
            threshold_passed,
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
        )
        kernel_us = (time.perf_counter_ns() - kernel_started) // 1000
        similarity = math.nan
        if norm_x and norm_y:
            similarity = dot / math.sqrt(norm_x * norm_y)
            similarity = max(-1.0, min(1.0, similarity))
        server_us = (time.perf_counter_ns() - started) // 1000
        return (
            bool(should_skip),
            bool(threshold_passed),
            int(skip_streak),
            similarity,
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

    def _run_ip(self, x_buffer, length, step_index, threshold_q15,
                warmup_steps, max_skips, reset_state):
        self._write_u64(X_LOW, x_buffer.physical_address)
        self.ip.write(LENGTH, length)
        self.ip.write(STEP_INDEX, step_index)
        self.ip.write(THRESHOLD_Q15, threshold_q15)
        self.ip.write(WARMUP_STEPS, warmup_steps)
        self.ip.write(MAX_CONSECUTIVE_SKIPS, max_skips)
        self.ip.write(RESET_STATE, reset_state)
        self.ip.write(CTRL, 0x01)
        deadline = time.monotonic() + IP_TIMEOUT_SECONDS
        while (self.ip.read(CTRL) & 0x2) == 0:
            if time.monotonic() >= deadline:
                raise HardwareTimeout("cosine_skip IP did not finish within 1 second")
        return (
            self.ip.read(AP_RETURN),
            self.ip.read(THRESHOLD_PASSED_OUT),
            self.ip.read(SKIP_STREAK_OUT),
            self._read_i64(DOT_OUT_LOW),
            self._read_u64(NORM_X_LOW),
            self._read_u64(NORM_Y_LOW),
        )


def send_response(connection, status, decision=0, threshold_passed=0,
                  skip_streak=0, step=0, kernel_us=0, server_us=0,
                  similarity=math.nan):
    connection.sendall(
        RESPONSE.pack(
            MAGIC,
            status,
            int(bool(decision)),
            int(bool(threshold_passed)),
            min(255, max(0, int(skip_streak))),
            int(step),
            int(kernel_us),
            int(server_us),
            float(similarity),
        )
    )


def serve_client(connection, address, hardware):
    print("Client connected:", address, flush=True)
    hardware.reset()
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.settimeout(15.0)
    while True:
        header = bytearray(REQUEST.size)
        recv_exact(connection, header)
        magic, command, step, length, threshold, warmup, max_skips = REQUEST.unpack(header)
        if magic != MAGIC:
            send_response(connection, STATUS_PROTOCOL_ERROR, step=step)
            return
        if command == CMD_PING:
            send_response(connection, STATUS_OK, step=step)
            continue
        if command == CMD_RESET:
            hardware.reset()
            send_response(connection, STATUS_OK, step=step)
            continue
        if command != CMD_STEP:
            send_response(connection, STATUS_PROTOCOL_ERROR, step=step)
            return
        if length <= 0 or length > MAX_VECTOR_LEN:
            send_response(connection, STATUS_LENGTH_ERROR, step=step)
            return

        hardware.receive_feature(connection, length)
        try:
            decision, threshold_passed, skip_streak, similarity, kernel_us, server_us = (
                hardware.decide(step, length, threshold, warmup, max_skips)
            )
        except HardwareTimeout as exc:
            print("FPGA timeout:", exc, flush=True)
            send_response(connection, STATUS_HARDWARE_ERROR, step=step)
            raise
        send_response(
            connection,
            STATUS_OK,
            decision,
            threshold_passed,
            skip_streak,
            step,
            kernel_us,
            server_us,
            similarity,
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
    print(f"PYNQ cosine server listening on {args.host}:{args.port}", flush=True)
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
