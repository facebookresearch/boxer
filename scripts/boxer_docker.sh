#!/bin/bash

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Usage: scripts/boxer_docker.sh [COMMAND [ARGS...]]

Enter the boxer Docker container (built from docker/Dockerfile).
Volumes for ckpts/, sample_data/, and output/ are mounted automatically

Examples:
  scripts/boxer_docker.sh
  scripts/boxer_docker.sh python run_boxer.py --input nym10_gen1 --skip_viz
  scripts/boxer_docker.sh bash tests/run_tests.sh --no-open

Note: X11 forwarding is granted automatically when DISPLAY is set and revoked on exit!
EOF
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

XHOST_GRANTED=false
if [ -n "$DISPLAY" ]; then
    if xhost +local:docker > /dev/null 2>&1; then
        XHOST_GRANTED=true
    fi
fi

docker compose -f "$SCRIPT_DIR/../docker/docker-compose.yml" run --rm --remove-orphans boxer "${@:-bash}"

if [ "$XHOST_GRANTED" = true ]; then
    xhost -local:docker > /dev/null 2>&1 || true
fi
