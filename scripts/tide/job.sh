#!/usr/bin/env bash
# Background jobs that survive closed browser tabs and kernel restarts (not a server stop).
#   bash scripts/tide/job.sh start E2 glass_crack_models.train.train_yolo --config e2_yolo_linear.yaml --seed 0
#   bash scripts/tide/job.sh status
#   bash scripts/tide/job.sh tail E2 [lines]
#   bash scripts/tide/job.sh stop E2
# Logs and pid files: $OUT_ROOT/logs/<name>.{log,pid}. Training scripts resume from their last
# checkpoint, so after a server restart run the same start command again.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LOGS="$OUT_ROOT/logs"
mkdir -p "$LOGS"

running() { [ -f "$LOGS/$1.pid" ] && kill -0 "$(cat "$LOGS/$1.pid")" 2>/dev/null; }
# Last lines of a log, with progress-bar carriage returns and terminal escape codes removed.
clean_tail() { tail -c 200000 "$LOGS/$1.log" 2>/dev/null | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -v '^\s*$' | tail -n "$2"; }

cmd="${1:-status}"; shift || true
case "$cmd" in
    start)
        name="$1"; shift
        if running "$name"; then
            echo "$name is already running (pid $(cat "$LOGS/$name.pid")); log: $LOGS/$name.log"; exit 1
        fi
        cd "$CODE_DIR"
        echo "=== $(date '+%F %T') | ${GPU_NAME:-no gpu} | workers $WORKERS | $*" >> "$LOGS/$name.log"
        setsid nohup "$PY" -u -m "$@" >> "$LOGS/$name.log" 2>&1 < /dev/null &
        echo $! > "$LOGS/$name.pid"
        echo "started $name (pid $!); log: $LOGS/$name.log"
        ;;
    status)
        shopt -s nullglob
        for p in "$LOGS"/*.pid; do
            n=$(basename "$p" .pid)
            state=$(running "$n" && echo running || echo stopped)
            last=$(clean_tail "$n" 1 | cut -c1-150)
            printf '%-14s %-8s %s\n' "$n" "$state" "$last"
        done
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
        ;;
    tail)
        clean_tail "$1" "${2:-30}"
        ;;
    stop)
        if running "$1"; then kill -- -"$(cat "$LOGS/$1.pid")" 2>/dev/null || kill "$(cat "$LOGS/$1.pid")"; echo "stopped $1"
        else echo "$1 is not running"; fi
        ;;
    *)
        echo "usage: job.sh start <name> <module> [args...] | status | tail <name> [lines] | stop <name>"; exit 2
        ;;
esac
