#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUN_USER=root
SERVICE_FILE=/etc/systemd/system/pynq-cosine.service
if [ -x /usr/local/share/pynq-venv/bin/python3 ]; then
    PYTHON_BIN=/usr/local/share/pynq-venv/bin/python3
else
    PYTHON_BIN=$(command -v python3)
fi

"$PYTHON_BIN" -c "import pynq" || {
    echo "The selected Python cannot import pynq: $PYTHON_BIN" >&2
    exit 1
}

sudo sh -c "cat > '$SERVICE_FILE'" <<EOF
[Unit]
Description=PYNQ cosine skip-decision server
After=network-online.target jupyter.service
Wants=network-online.target jupyter.service

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$SCRIPT_DIR
ExecStartPre=/bin/sleep 5
ExecStart=/bin/bash $SCRIPT_DIR/start_pynq_cosine.sh
Restart=on-failure
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable pynq-cosine.service
sudo systemctl restart pynq-cosine.service
sudo systemctl --no-pager status pynq-cosine.service
