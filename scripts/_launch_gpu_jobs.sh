#!/bin/bash
# Launch TideTail GPU jobs — each as a truly independent session.
# Usage: bash scripts/_launch_gpu_jobs.sh [standard_fm|vae|timegrad|all]
# PIDs are written to /tmp/tidetail_gpu_pids.txt

set -e
cd /data/yingda/TideTail

launch_job() {
    local name=$1
    local gpu=$2
    local cmd=$3
    local log=/tmp/${name}.log
    
    # Create independent session: setsid + detach from stdin/stdout
    setsid env CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" \
        < /dev/null >> "$log" 2>&1 &
    local pid=$!
    
    # Write PID with timestamp
    echo "$(date -Iseconds) $name PID=$pid GPU=$gpu" >> /tmp/tidetail_gpu_pids.txt
    echo "✅ $name launched PID=$pid GPU=$gpu → $log"
}

case "$1" in
    standard_fm|all)
        launch_job "standard_fm_a3" 4 \
            'python3 scripts/run_a3_flowmatching_augment.py \
                --regions ny_bight us_gulf_coast south_atlantic_charleston southeast_australia thames_north_sea \
                --fm-seed 42 --seeds 42 99 --augment-ratios 0.1 0.25 0.5 --target-steps 8000 \
                --fm-prefix flow_matching'
        ;;
    vae|all)
        launch_job "vae_full" 1 \
            'python3 scripts/train_gen_bench.py --model vae \
                --regions ny_bight us_gulf_coast south_atlantic_charleston southeast_australia thames_north_sea \
                --seeds 42 99 --epochs 100 --batch-size 256 --lr 0.0001 --max-samples 5000'
        ;;
    timegrad|all)
        launch_job "timegrad_full" 3 \
            'python3 scripts/train_gen_bench.py --model timegrad \
                --regions ny_bight us_gulf_coast south_atlantic_charleston southeast_australia thames_north_sea \
                --seeds 42 99 --epochs 50 --batch-size 256 --lr 0.0001 --max-samples 5000'
        ;;
    *)
        echo "Usage: $0 [standard_fm|vae|timegrad|all]"
        exit 1
        ;;
esac
