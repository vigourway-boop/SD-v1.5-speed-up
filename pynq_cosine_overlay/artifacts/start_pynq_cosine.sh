#!/bin/bash
set -e

set -a
. /etc/environment
set +a
for profile_script in /etc/profile.d/*.sh; do
    source "$profile_script"
done

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /usr/local/share/pynq-venv/bin/python3 \
    "$SCRIPT_DIR/pynq_cosine_server.py" \
    --bitstream "$SCRIPT_DIR/cosine_overlay.bit" \
    --port 9000
