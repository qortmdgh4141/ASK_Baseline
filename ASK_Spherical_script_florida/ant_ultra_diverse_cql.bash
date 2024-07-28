#!/bin/bash
source /home/qortmdgh4141/anaconda3/etc/profile.d/conda.sh
conda activate ask_hilp

script_dir=$(dirname "$(realpath "$0")") # 현재 스크립트의 절대 경로를 구함
bash_script_path="${script_dir}/experiment_output" # experiment_output 경로를 구성
parent_dir=$(dirname "$script_dir") # 현재 스크립트의 부모 디렉토리를 구함
main_py_path="/home/qortmdgh4141/disk/HIQL_Team_Project/TG/main.py" # 부모 디렉토리에서 main.py 파일의 경로를 구성

# current_time="03-03_09:23"
current_time=$(date "+%m-%d_%H:%M")
project="cql-ant-ultra"

extra_name="cql-A_50_q_32dim_eval"

gpu="1"
start=0
end_=1

use_rep=''                     # ["hiql_goal_encoder", "hilp_subgoal_encoder", "hilp_encoder", "vae_encoder"]
kl_loss=0
final_goal=0
mse_loss=0.0
correction_value=0
high_action_in_hilp=0
low_actor_train_with_high_actor=0
n_step_hilp=0
high_temperature=1
temperature=1
discount=0.99
pretrain_expectile=0.95
distance='first'
key_node_q=0
key_node_train=0

for ((i=start; i<=end_; i++)); do
    python ${main_py_path} --gpu ${gpu} --save_dir ${bash_script_path} --run_group ant_ultra_diverse_${extra_name}_${current_time} --env_name antmaze-ultra-diverse-v0 --project ${project} --seed $((i*100)) --algo_name cql --pretrain_steps 1000002 --eval_interval 50000 --save_interval 100000 --eval_episodes 24 --num_video_episodes 6 --way_steps 50 --sparse_data 0 --expert_data_On 0 \
    --spherical_On 0.0 --rep_type concat \
    --mapping_threshold 0.0 --rep_normalizing_On 1 --rep_dim 10 --hilp_skill_dim 32 --build_keynode_time during_training --keynode_num 300 --kmean_weight_On 0 --use_goal_info_On 0 --kmean_weight_type rtg_uniform --specific_dim_On 1 --keynode_ratio 0.0 --use_keynode_in_eval_On 1 \
    --relative_dist_in_eval_On 0 --mapping_method nearest --kl_loss ${kl_loss} --final_goal ${final_goal} --mse_loss ${mse_loss} --high_action_in_hilp ${high_action_in_hilp} --low_actor_train_with_high_actor ${low_actor_train_with_high_actor} --correction_value ${correction_value} --n_step_hilp ${n_step_hilp} --high_temperature ${high_temperature} --temperature ${temperature} --discount ${discount} --pretrain_expectile ${pretrain_expectile} --distance ${distance} --key_node_q ${key_node_q} --key_node_train ${key_node_train}
done


