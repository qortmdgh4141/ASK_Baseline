#!/bin/bash
source /home/qortmdgh4141/anaconda3/etc/profile.d/conda.sh
conda activate ask_hilp

script_dir=$(dirname "$(realpath "$0")") # 현재 스크립트의 절대 경로를 구함
bash_script_path="${script_dir}/experiment_output" # experiment_output 경로를 구성
parent_dir=$(dirname "$script_dir") # 현재 스크립트의 부모 디렉토리를 구함
main_py_path="$parent_dir/main.py" # 부모 디렉토리에서 main.py 파일의 경로를 구성

# current_time="03-03_09:23"
current_time="06-10_20:40"
# current_time=$(date "+%m-%d_%H:%M")
project="Spherical_Test"

extra_name="RTG_Hilp_Uniform_Spherical_On"

gpu="0"
start=1
end_=3

for ((i=start; i<=end_; i++)); do
    python ${main_py_path} --gpu ${gpu} --save_dir ${bash_script_path} --run_group ant_ultra_diverse_${extra_name}_${current_time} --env_name antmaze-ultra-diverse-v0 --project ${project} \
    --seed $((i*100)) --pretrain_steps 1000002 --eval_interval 100000 --save_interval 100000 --eval_episodes 30 --num_video_episodes 2 --way_steps 35 \
    --sparse_data 0 --expert_data_On 0 \
    --spherical_On 1 \
    --use_rep hiql_goal_encoder --rep_normalizing_On 1 --rep_dim 10 --hilp_skill_dim 0 --keynode_dim 10 \
    --build_keynode_time during_training --keynode_num 1000 --kmean_weight_On 1 --use_goal_info_On 1 --kmean_weight_type rtg_uniform --specific_dim_On 0 --keynode_ratio 0.0 --use_keynode_in_eval_On 1 \
    --relative_dist_in_eval_On 1 --mapping_method triple
done
