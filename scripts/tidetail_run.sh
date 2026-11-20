#!/bin/bash
# TideTail persistent experiment runner — Wave 2.8 P0-2 artifact protection.
#
# Wraps one experiment command so that:
#   * every run gets an immutable per-run directory with logs, status,
#     exit code and artifact checksums;
#   * any pre-existing output artifact is backed up BEFORE the command
#     can overwrite it;
#   * a persistent JSONL manifest records every run (durable storage,
#     outside git);
#   * the command itself runs detached from the calling shell, so it is
#     safe to launch under `systemd-run --user` for session-independent
#     execution (P0-1).
#
# Usage:
#   scripts/tidetail_run.sh <name> <output_artifact_relative_path> <cmd...>
#
# Recommended launch (survives TRAE session resets; linger enabled):
#   export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1018/bus
#   systemd-run --user --unit=tidetail-<name> \
#       --working-directory=/data/yingda/TideTail \
#       scripts/tidetail_run.sh <name> <artifact> <cmd...>
#
# Output layout:
#   data/results/runs/<run_id>/            immutable per-run directory
#       cmd.txt env.txt stdout.log status.json <artifact copy>
#   data/results/backups/<run_id>/         pre-overwrite backup
#   data/results/RUN_MANIFEST.jsonl        append-only manifest

set -u
ROOT="/data/yingda/TideTail"

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <name> <output_artifact_relpath> <cmd...>" >&2
    exit 64
fi

NAME="$1"; shift
ARTIFACT="$1"; shift
CMD=("$@")

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
CMDHASH=$(printf '%s\n' "${CMD[@]}" | sha256sum | cut -c1-8)
RUN_ID="${NAME}_${STAMP}_${CMDHASH}"
RUN_DIR="$ROOT/data/results/runs/$RUN_ID"
BACKUP_DIR="$ROOT/data/results/backups/$RUN_ID"
MANIFEST="$ROOT/data/results/RUN_MANIFEST.jsonl"

mkdir -p "$RUN_DIR" "$BACKUP_DIR"

printf '%s\n' "${CMD[@]}" > "$RUN_DIR/cmd.txt"
{
    echo "run_id=$RUN_ID"
    echo "git_sha=$(git -C $ROOT rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty=$(git -C $ROOT status --porcelain 2>/dev/null | wc -l)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RUN_DIR/env.txt"

# --- backup-before-overwrite (P0-2 acceptance) ---
ART_ABS="$ROOT/$ARTIFACT"
if [ -f "$ART_ABS" ]; then
    cp -p "$ART_ABS" "$BACKUP_DIR/$(basename "$ARTIFACT")"
    echo "[tidetail_run] backed up existing $ARTIFACT -> $BACKUP_DIR/"
fi

echo "[tidetail_run] run_id=$RUN_ID"
echo "[tidetail_run] cmd: ${CMD[*]}"
echo "[tidetail_run] log:  $RUN_DIR/stdout.log"

START=$(date +%s)
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/stdout.log"
EXIT=${PIPESTATUS[0]}
END=$(date +%s)

# --- immutable artifact copy + checksum ---
if [ -f "$ART_ABS" ]; then
    cp -p "$ART_ABS" "$RUN_DIR/$(basename "$ARTIFACT")"
    SHA=$(sha256sum "$ART_ABS" | cut -d' ' -f1)
else
    SHA=""
fi

python3 - "$RUN_DIR" "$MANIFEST" "$RUN_ID" "$NAME" "$ARTIFACT" \
        "$EXIT" "$START" "$END" "$SHA" <<'PYEOF'
import json, sys, pathlib
run_dir, manifest, run_id, name, artifact, exit_code, start, end, sha = sys.argv[1:10]
status = {
    "run_id": run_id, "name": name, "artifact": artifact,
    "exit_code": int(exit_code), "started_epoch": int(start),
    "ended_epoch": int(end), "elapsed_s": int(end) - int(start),
    "artifact_sha256": sha or None,
}
pathlib.Path(run_dir, "status.json").write_text(json.dumps(status, indent=2))
with open(manifest, "a") as f:
    f.write(json.dumps(status) + "\n")
PYEOF

echo "[tidetail_run] exit_code=$EXIT elapsed=$((END-START))s sha256=$SHA"
exit "$EXIT"
