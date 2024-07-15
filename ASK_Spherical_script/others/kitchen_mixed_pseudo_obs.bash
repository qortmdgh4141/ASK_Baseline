#!/bin/bash
source /home/tako/anaconda3/etc/profile.d/conda.sh
conda activate HI

script_dir=$(dirname "$(realpath "$0")")                # 현재 스크립트의 절대 경로를 구함
bash_script_path="${script_dir}/experiment_output"      # experiment_output 경로를 구성
parent_dir=$(dirname "$script_dir")                     # 현재 스크립트의 부모 디렉토리를 구함
main_py_path="/data/tg_park/study/ASK_Baseline/main.py" # 부모 디렉토리에서 main.py 파일의 경로를 구성

# current_time="03-03_09:23"
current_time=$(date "+%m-%d_%H:%M")
# current_time="07-10_10:50"
project="kitchen"

extra_name="4kcn_id_5_.95_g_4_.5"

gpu="2"
start=0
end_=2

use_rep='' # ["hiql_goal_encoder", "hilp_subgoal_encoder", "hilp_encoder", "vae_encoder"]
pseudo_obs=0.5
high_temperature=1
alpha=0.95
gamma=4
identify='two'
identify_hidden_dim=512
conditioned=1
negative_train=1
kl_loss=1
splite_actor=0
identify_num_layers=5


for ((i = start; i <= end_; i++)); do
    python ${main_py_path} --gpu ${gpu} --save_dir ${bash_script_path} --run_group Kitchen_${extra_name}_${current_time} --env_name kitchen-mixed-v0 --project ${project} \
        --seed $((i * 100)) --pretrain_steps 500002 --eval_interval 50000 --save_interval 100000 --eval_episodes 6 --num_video_episodes 24 --algo_name ask --way_steps 25 \
        --sparse_data 0 --expert_data_On 0 \
        --spherical_On 0.0 --rep_type concat \
        --mapping_threshold 0.0 \
        --use_rep ${use_rep} --rep_normalizing_On 1 --rep_dim 10 --hilp_skill_dim 0 --keynode_dim 10 \
        --build_keynode_time during_training --keynode_num 300 --kmean_weight_On 1 --use_goal_info_On 0 --kmean_weight_type rtg_uniform --specific_dim_On 0 --keynode_ratio 0.0 --use_keynode_in_eval_On 0 \
        --relative_dist_in_eval_On 0 --mapping_method nearest --pseudo_obs ${pseudo_obs} --alpha ${alpha} --gamma ${gamma} --identify ${identify} --identify_hidden_dim ${identify_hidden_dim} --conditioned ${conditioned} --negative_train ${negative_train} --kl_loss ${kl_loss} --splite_actor ${splite_actor} --identify_num_layers ${identify_num_layers}
        # --high_temperature ${high_temperature}

done

# --model_base_path ${model_base_path} \ 
