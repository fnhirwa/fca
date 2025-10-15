#!/usr/bin/env bash
# Source this to expose vendored Prophet on PYTHONPATH
# Usage: source scripts/dev_env.sh

REPO_ROOT=$(cd "$(dirname "$0")"/.. && pwd)

export PYTHONPATH="$REPO_ROOT/third_party/prophet:$PYTHONPATH"
echo "PYTHONPATH updated to include: $REPO_ROOT/third_party/prophet"
