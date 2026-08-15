"""Shared CSK4 wire protocol for the PC client and PYNQ-Z2 server."""

import struct


MAGIC = b"CSK4"

CMD_RESET = 1
CMD_PING = 2
CMD_STEP = 3
CMD_CONFIG = 4

STATUS_OK = 0
STATUS_PROTOCOL_ERROR = 1
STATUS_LENGTH_ERROR = 2
STATUS_HARDWARE_ERROR = 3
STATUS_NOT_CONFIGURED = 4
STATUS_CONFIG_ERROR = 5

PHASE_FIXED = 0
PHASE_WARMUP = 1
PHASE_MIDDLE = 2
PHASE_LATE_ADAPTIVE = 3
PHASE_NONE = 255

PHASE_NAMES = {
    PHASE_FIXED: "fixed",
    PHASE_WARMUP: "warmup",
    PHASE_MIDDLE: "middle",
    PHASE_LATE_ADAPTIVE: "late_adaptive",
    PHASE_NONE: "none",
}
PHASE_CODES = {name: code for code, name in PHASE_NAMES.items()}

# Every command starts with command, step index, and optional byte length.
REQUEST = struct.Struct("!4sB3xII")

# enabled, total_steps, warmup_steps, max_skips, then nine float settings.
CONFIG = struct.Struct("!B3xIII9f")

# status/flags, phase, step/timings/thresholds, and controller statistics.
RESPONSE = struct.Struct("!4sBBBBBB2xIIIIIdddd")


def phase_name(code):
    return PHASE_NAMES.get(int(code), "unknown")


def phase_code(name):
    try:
        return PHASE_CODES[name]
    except KeyError:
        raise ValueError("Unknown threshold phase: {!r}".format(name))
