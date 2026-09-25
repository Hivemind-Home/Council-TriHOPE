#!/usr/bin/env bash
# Record exactly what code/config/environment produced the live runs (paper T1).
# The live campaign did not record its commit, and its thresholds were a
# box-local edit of configs/stream_small.yaml (CAMPAIGN_NOTES §10). Run this
# on the GPU box from the checkout that ran the campaign, BEFORE switching
# branches (it works from the current directory, so it can be streamed from
# the branch without checking it out):
#
#   git fetch origin campaign/iclr-2027-final-batch
#   git show origin/campaign/iclr-2027-final-batch:scripts/provenance_snapshot.sh | bash
#   # -> provenance/<timestamp>/
#
# Commit the resulting directory (it is small, and contains no secrets).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
out="provenance/$(date -u +%Y%m%dT%H%MZ)"
mkdir -p "$out"
{
  echo "host: $(hostname)"; echo "pwd: $(pwd)"; echo "date_utc: $(date -u)"
  echo "git_head: $(git rev-parse HEAD 2>/dev/null)"
  echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "git_describe: $(git describe --always --dirty 2>/dev/null)"
} > "$out/git.txt"
git status --porcelain=v1 > "$out/git_status.txt" 2>&1
git diff > "$out/git_diff.patch" 2>&1
git diff --stat > "$out/git_diff_stat.txt" 2>&1
git log --oneline -30 > "$out/git_log.txt" 2>&1
for f in configs/stream_small.yaml configs/stream_small.yaml.*bak*; do
  [ -f "$f" ] && cp "$f" "$out/$(basename "$f")"
done
[ -f configs/stream_small.yaml.pre_live_stop1.bak ] && \
  diff configs/stream_small.yaml.pre_live_stop1.bak configs/stream_small.yaml > "$out/stream_small_live_vs_prelive.diff"
python - > "$out/env.txt" 2>&1 <<'PY'
import platform, sys
print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
except Exception as e:  # noqa: BLE001
    print("torch import failed:", e)
for mod in ("transformers", "hydra", "omegaconf", "datasets", "numpy"):
    try:
        m = __import__(mod); print(mod, getattr(m, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print(mod, "missing", e)
PY
pip freeze > "$out/pip_freeze.txt" 2>&1
nvidia-smi > "$out/nvidia_smi.txt" 2>&1 || echo "nvidia-smi failed (NVML mismatch noted in CAMPAIGN_NOTES §3)" >> "$out/nvidia_smi.txt"
# the forced-merge gate the final batch relies on (periodic_merge uses the plateau path instead)
grep -n "if controller_enabled and step in force_steps" src/hivemind/training.py > "$out/force_gate_grep.txt" 2>&1
echo "wrote $out"
