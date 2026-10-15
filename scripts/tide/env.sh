# Source me: paths and per-session settings for the SDSU TIDE JupyterHub.
# Only $HOME persists across server restarts, so code, data, the Python environment,
# weight caches and outputs all live under $GC_ROOT.

export GC_ROOT="${GC_ROOT:-$HOME/glass_crack}"
export CODE_DIR="$GC_ROOT/cracked"
export DATA_ROOT="$GC_ROOT/bundle"
export OUT_ROOT="$GC_ROOT/outputs"
export VENV="$GC_ROOT/venv"
export PY="$VENV/bin/python"
export TORCH_HOME="$GC_ROOT/cache/torch"
export HF_HOME="$GC_ROOT/cache/huggingface"
export YOLO_CONFIG_DIR="$GC_ROOT/cache/ultralytics"
export PIP_CACHE_DIR="$GC_ROOT/cache/pip"

# CPU quota of this server (cgroup v2, then v1); nproc reports the whole node inside Kubernetes.
_gc_cpus=""
if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r _q _p < /sys/fs/cgroup/cpu.max
    [ "$_q" != "max" ] && _gc_cpus=$(( (_q + _p - 1) / _p ))
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    _q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); _p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$_q" -gt 0 ] && _gc_cpus=$(( (_q + _p - 1) / _p ))
fi
export GC_CPUS="${_gc_cpus:-$(nproc)}"
export GC_SHM_MB="$(df -Pm /dev/shm 2>/dev/null | awk 'NR==2 {print $2}')"

# DataLoader workers pass batches through /dev/shm; with the 64 MB container default they crash.
if [ -z "${WORKERS:-}" ]; then
    if [ "${GC_SHM_MB:-0}" -lt 1024 ]; then
        export WORKERS=0
    else
        _w=$(( GC_CPUS - 1 )); [ "$_w" -gt 8 ] && _w=8; [ "$_w" -lt 0 ] && _w=0
        export WORKERS="$_w"
    fi
fi
export GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
unset _gc_cpus _q _p _w
