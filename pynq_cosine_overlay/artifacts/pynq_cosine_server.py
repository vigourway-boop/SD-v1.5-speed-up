#!/usr/bin/env python3
"""PYNQ-Z2 TCP service for hardware cosine skip decisions."""

import argparse
import math
import socket
import struct
import time

import numpy as np
from pynq import Overlay, allocate


MAGIC = b"CSK2"
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
Y_LOW = 0x24
LENGTH = 0x30
THRESHOLD_Q15 = 0x38
DOT_OUT_LOW = 0x40
NORM_X_LOW = 0x58
NORM_Y_LOW = 0x70
MAX_VECTOR_LEN = 32768
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
        self.incoming = allocate(shape=(MAX_VECTOR_LEN,), dtype=np.int16)
        self.reference = allocate(shape=(MAX_VECTOR_LEN,), dtype=np.int16)
        self.has_reference = False
        self.reference_length = 0
        self.consecutive_skips = 0

    def close(self):
        self.incoming.freebuffer()
        self.reference.freebuffer()

    def reset(self):
        self.has_reference = False
        self.reference_length = 0
        self.consecutive_skips = 0

    def receive_feature(self, connection, length):
        target = memoryview(self.incoming).cast("B")[:length * 2]
        recv_exact(connection, target)
        self.incoming.flush()

    def decide(self, step_index, length, threshold_q15, warmup_steps, max_skips):
        started = time.perf_counter_ns()
        threshold_passed = False
        similarity = math.nan
        kernel_us = 0
        if self.has_reference and self.reference_length == length:
            kernel_started = time.perf_counter_ns()
            threshold_result, dot, norm_x, norm_y = self._run_ip(
                self.incoming, self.reference, length, threshold_q15
            )
            threshold_passed = bool(threshold_result)
            if norm_x and norm_y:
                similarity = dot / math.sqrt(norm_x * norm_y)
                similarity = max(-1.0, min(1.0, similarity))
            kernel_us = (time.perf_counter_ns() - kernel_started) // 1000

        should_skip = (
            self.has_reference
            and threshold_passed
            and step_index >= warmup_steps
            and self.consecutive_skips < max_skips
        )
        if should_skip:
            self.consecutive_skips += 1
        else:
            self.incoming, self.reference = self.reference, self.incoming
            self.has_reference = True
            self.reference_length = length
            self.consecutive_skips = 0
        server_us = (time.perf_counter_ns() - started) // 1000
        return should_skip, threshold_passed, similarity, kernel_us, server_us

    def _write_u64(self, offset_low, value):
        value = int(value)
        self.ip.write(offset_low, value & 0xFFFFFFFF)
        self.ip.write(offset_low + 4, (value >> 32) & 0xFFFFFFFF)

    def _read_u64(self, offset_low):
        return self.ip.read(offset_low) | (self.ip.read(offset_low + 4) << 32)

    def _read_i64(self, offset_low):
        value = self._read_u64(offset_low)
        return value - (1 << 64) if value & (1 << 63) else value

    def _run_ip(self, x_buffer, y_buffer, length, threshold_q15):
        self._write_u64(X_LOW, x_buffer.physical_address)
        self._write_u64(Y_LOW, y_buffer.physical_address)
        self.ip.write(LENGTH, length)
        self.ip.write(THRESHOLD_Q15, threshold_q15)
        self.ip.write(CTRL, 0x01)
        deadline = time.monotonic() + IP_TIMEOUT_SECONDS
        while (self.ip.read(CTRL) & 0x2) == 0:
            if time.monotonic() >= deadline:
                raise HardwareTimeout("cosine_skip IP did not finish within 1 second")
        return (
            self.ip.read(AP_RETURN),
            self._read_i64(DOT_OUT_LOW),
            self._read_u64(NORM_X_LOW),
            self._read_u64(NORM_Y_LOW),
        )


def send_response(connection, status, decision=0, threshold_passed=0, step=0,
                  kernel_us=0, server_us=0, similarity=math.nan):
    connection.sendall(
        RESPONSE.pack(
            MAGIC,
            status,
            int(bool(decision)),
            int(bool(threshold_passed)),
            0,
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
            decision, threshold_passed, similarity, kernel_us, server_us = (
                hardware.decide(step, length, threshold, warmup, max_skips)
            )
        except HardwareTimeout as exc:
            print("FPGA timeout:", exc, flush=True)
            send_response(connection, STATUS_HARDWARE_ERROR, step=step)
            hardware.reset()
            return
        send_response(
            connection,
            STATUS_OK,
            decision,
            threshold_passed,
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
