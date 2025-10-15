#!/usr/bin/env bash
set -euo pipefail

# Sync vendored Prophet (MILVLG/prophet) via git subtree
# Usage:
#   scripts/vendor_sync_prophet.sh            # pulls 'main'
#   scripts/vendor_sync_prophet.sh <ref>      # pulls specific branch or tag
#

REPO_ROOT=$(cd "$(dirname "$0")"/.. && pwd)
cd "$REPO_ROOT"

REF=${1:-main}
REMOTE_URL=${REMOTE_URL:-https://github.com/MILVLG/prophet}
PREFIX=third_party/prophet

if [ ! -d "$PREFIX" ]; then
  echo "[vendor] Adding Prophet at $PREFIX (ref=$REF)"
  git subtree add --prefix "$PREFIX" "$REMOTE_URL" "$REF" --squash
else
  echo "[vendor] Pulling updates for Prophet at $PREFIX (ref=$REF)"
  git subtree pull --prefix "$PREFIX" "$REMOTE_URL" "$REF" --squash
fi

echo "[vendor] Done."
