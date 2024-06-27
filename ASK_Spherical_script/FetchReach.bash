#!/bin/bash
source /data/tg_park/study/miniforge3/etc/profile.d/conda.sh
conda activate HI_fetch

script_dir=$(dirname "$(realpath "$0")")                # 현재 스크립트의 절대 경로를 구함
bash_script_path="${script_dir}/experiment_output"      # experiment_output 경로를 구성
parent_dir=$(dirname "$script_dir")                     # 현재 스크립트의 부모 디렉토리를 구함
main_py_path="/data/tg_park/study/ASK_Baseline/main.py" # 부모 디렉토리에서 main.py 파일의 경로를 구성

# current_time="03-03_09:23"
current_time=$(date "+%m-%d_%H:%M")
project="FetchReach"

extra_name="hiql_expert"

gpu="0"
start=0
end_=0

value_function_num='flat'   # 'flat', 'hierarchy'
low_dim_clustering="" # [tsne_dim, pca_dim, ''] ex) hilp_2 : hilp - 2dim, pca_2 : pca - 2dim

# model_base_path="/home/qortmdgh4141/disk/HIQL_Team_Project/ASK/ASK_Baseline_spherical_goal_only/ASK_Spherical_goal_only_script/experiment_output/ant_ultra_diverse_RTG_Hilp_Uniform_Goal_Only_06-10_21:55"
# model_base_path="/home/qortmdgh4141/disk/HIQL_Team_Project/ASK/ASK_Baseline_spherical_goal_only/ASK_Spherical_goal_only_script/experiment_output/ant_ultra_diverse_RTG_Hilp_Uniform_Goal_Only_Concat_06-10_21:55"

for ((i = start; i <= end_; i++)); do
    python ${main_py_path} --gpu ${gpu} --save_dir ${bash_script_path} --run_group FetchReach_${extra_name}_${current_time} --env_name FetchReach-v1 --project ${project} \
        --seed $((i * 100)) --pretrain_steps 500002 --eval_interval 25000 --save_interval 100000 --eval_episodes 6 --num_video_episodes 24 --way_steps 5 \
        --sparse_data 0 --expert_data_On 0 \
        --spherical_On 1 --rep_type state \
        --mapping_threshold 99 \
        --use_rep 0 --rep_normalizing_On 1 --rep_dim 10 --hilp_skill_dim 0 --keynode_dim 10 \
        --build_keynode_time during_training --keynode_num 300 --kmean_weight_On 1 --use_goal_info_On 0 --kmean_weight_type rtg_uniform --specific_dim_On 0 --keynode_ratio 0.0 --use_keynode_in_eval_On 0 \
        --relative_dist_in_eval_On 1 --mapping_method nearest \
        --value_function_num ${value_function_num}
    # --low_dim_clustering ${low_dim_clustering}

done
