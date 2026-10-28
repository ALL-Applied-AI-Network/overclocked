#!/usr/bin/env bash
# Copy the Isaac cell server to the Spark's fanuc_er4ia_sim folder (SFTP can't follow the /workspace symlink).
set -euo pipefail
cd "$(dirname "$0")/.."
SPARK=yt-overclocked728b@172.30.102.93
DEST=/home/yt-overclocked728b/workspace/saai/fanuc_er4ia_sim
scp -q sim/isaac_cell.py mock_cell.py "$SPARK:$DEST/"
scp -qr calibration "$SPARK:$DEST/"
echo "copied to $DEST"
