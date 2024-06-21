#!/bin/bash
source /home/tako/anaconda3/etc/profile.d/conda.sh
conda activate HI

current_time=$(date "+%m-%d_%H:%M")
gpu="1"

cores=1
start=1
end=start+cores
for ((i = start; i <= end; i++)); do
    python /home/spectrum/study/ASK_Baseline/dataset_render.py --cores ${cores} --iter ${i} &
done
