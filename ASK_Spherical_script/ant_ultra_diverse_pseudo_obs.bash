#!/bin/bash
source /home/tako/anaconda3/etc/profile.d/conda.sh
conda activate HI

script_dir=$(dirname "$(realpath "$0")")                # 현재 스크립트의 절대 경로를 구함
bash_script_path="${script_dir}/experiment_output"      # experiment_output 경로를 구성
parent_dir=$(dirname "$script_dir")                     # 현재 스크립트의 부모 디렉토리를 구함
main_py_path="/data/tg_park/study/ASK_Baseline/main.py" # 부모 디렉토리에서 main.py 파일의 경로를 구성

# current_time="03-03_09:23"
current_time=$(date "+%m-%d_%H:%M")
project="ant-ultra"

extra_name="pseudo_100"

gpu="1"
start=0
end_=3

use_rep=0 # ["hiql_goal_encoder", "hilp_subgoal_encoder", "hilp_encoder", "vae_encoder"]
pseudo_obs=100
high_temperature=1

for ((i = start; i <= end_; i++)); do
    python ${main_py_path} --gpu ${gpu} --save_dir ${bash_script_path} --run_group ant_ultra_diverse_${extra_name}_${current_time} --env_name antmaze-ultra-diverse-v0 --project ${project} \
        --seed $((i * 100)) --pretrain_steps 1000002 --eval_interval 100000 --save_interval 100000 --eval_episodes 6 --num_video_episodes 24 --way_steps 35 \
        --sparse_data 0 --expert_data_On 0 \
        --spherical_On 0.0 --rep_type concat \
        --mapping_threshold 0.0 \
        --use_rep ${use_rep} --rep_normalizing_On 1 --rep_dim 10 --hilp_skill_dim 0 --keynode_dim 10 \
        --build_keynode_time during_training --keynode_num 300 --kmean_weight_On 1 --use_goal_info_On 0 --kmean_weight_type rtg_uniform --specific_dim_On 0 --keynode_ratio 0.0 --use_keynode_in_eval_On 0 \
        --relative_dist_in_eval_On 0 --mapping_method nearest --pseudo_obs ${pseudo_obs} --high_temperature ${high_temperature}

done

# --model_base_path ${model_base_path} \