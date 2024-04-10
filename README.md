# ASK: Adjusting Subgoals via Key Nodes for Offline Hierarchical Reinforcement Learning

## Installation

```
conda create --name ask python=3.8
conda activate ask
pip install -r requirements.txt --no-deps
pip install "jax[cuda11_cudnn82]==0.4.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
## Script

```
# ASK on antmaze-ultra-diverse
python main.py --save_dir "experiment_output" --run_group "ant-ultra-diverse"  --seed 0 --env_name antmaze-ultra-diverse-v0 --pretrain_steps 1000002 --eval_interval 100000 --save_interval 100000 --algo_name ASK --way_steps 35 --use_keynode_in_train_On 1 --weight 1 --node 225 --keynode_ratio0.75 --use_keynode_in_eval_On 1 --gpu "0"


# ASK on antmaze-ultra-play
python main.py --save_dir "experiment_output" --run_group "ant-ultra-play" --seed 0 --env_name antmaze-ultra-play-v0 --pretrain_steps 1000002 --eval_interval 100000 --save_interval 100000 --algo_name ASK --way_steps 35 --use_keynode_in_train_On 1 --weight 1 --node 225 --keynode_ratio0.75 --use_keynode_in_eval_On 1 --gpu "0"


# ASK on antmaze-large-diverse
python main.py --save_dir "experiment_output" --run_group "ant-large-diverse" --seed 0 --env_name antmaze-large-diverse-v2 --pretrain_steps 500002 --eval_interval 100000 --save_interval 100000 --algo_name ASK --way_steps 25 --use_keynode_in_train_On 1 --weight 1 --node 180 --keynode_ratio0.75 --use_keynode_in_eval_On 1 --gpu "0"


# ASK on antmaze-large-play
python main.py --save_dir "experiment_output" --run_group "ant-large-play" --seed 0 --env_name antmaze-large-play-v2 --pretrain_steps 500002 --eval_interval 100000 --save_interval 100000 --algo_name ASK --way_steps 25 --use_keynode_in_train_On 1 --weight 1 --node 180 --keynode_ratio0.75 --use_keynode_in_eval_On 1 --gpu "0"


# ASK on antmaze-medium-diverse
python main.py --save_dir "experiment_output" --run_group "ant-medium-diverse" --seed 0 --env_name antmaze-medium-diverse-v2 --pretrain_steps 500002 --eval_interval 100000 --save_interval 100000 --algo_name ASK --way_steps 25 --use_keynode_in_train_On 1 --weight 1 --node 180 --keynode_ratio0.75 --use_keynode_in_eval_On 1 --gpu "0"


# ASK on antmaze-medium-play
python main.py --save_dir "experiment_output" --run_group "ant-medium-play" --seed 0 --env_name antmaze-medium-play-v2 --pretrain_steps 500002 --eval_interval 100000 --save_interval 100000 --algo_name ASK --way_steps 25 --use_keynode_in_train_On 1 --weight 1 --node 180 --keynode_ratio0.75 --use_keynode_in_eval_On 1 --gpu "0"


# ASK on kitchen-partial
python main.py --save_dir "experiment_output" --run_group "kitchen-partial" --seed 0 --env_name kitchen-partial-v0 --pretrain_steps 500002 --eval_interval 25000 --save_interval 25000 --algo_name ASK --way_steps 25 --use_keynode_in_train_On 1 --weight 1 --node 300 --keynode_ratio0.75 --use_keynode_in_eval_On 1 --gpu "0"


# ASK on kitchen-mixed
python main.py --save_dir "experiment_output" --run_group "kitchen-mixed" --seed 0 --env_name kitchen-mixed-v0 --pretrain_steps 500002 --eval_interval 25000 --save_interval 25000 --algo_name ASK --way_steps 25 --use_keynode_in_train_On 1 --weight 1 --node 300 --keynode_ratio0.75 --use_keynode_in_eval_On 1 --gpu "0"
```

## License
MIT
