#!/bin/bash
set -euo pipefail
window_dir="$(cd "$(dirname "$0")" && pwd -P)"
/opt/homebrew/bin/python3 "$window_dir/queue-check.py"
sudo /opt/homebrew/bin/python3 -I "$window_dir/admin-window.py" start
