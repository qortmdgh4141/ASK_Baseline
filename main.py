import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
gpu_index = sys.argv[sys.argv.index('--gpu') + 1] if '--gpu' in sys.argv else "2" # Default to GPU 0 if no --gpu argument
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
print("Using GPU: ", gpu_index)

project = sys.argv[sys.argv.index('--project') + 1] if '--project' in sys.argv else "test" 

import jax
import flax
import gzip
import tqdm
import time
import wandb
import pickle
import datetime
import numpy as np
import jax.numpy as jnp
import tensorflow as tf

from absl import app, flags
from functools import partial
from src.gc_dataset import GCSDataset
from src.agents import ask as learner
from ml_collections import config_flags
from src.utils import record_video, CsvLogger, plot_value_map, plot_q_map, plot_q_map_modified, plot_value_map_others
from jaxrl_m.wandb import setup_wandb, default_wandb_config
from src import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils, keynode_utils
from src.d4rl_utils import plot_obs
from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories, EpisodeMonitor

FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', f'experiment_output/', '')
flags.DEFINE_string('run_group', 'EXP', '')
flags.DEFINE_string('env_name', 'antmaze-ultra-diverse-v0', '')
flags.DEFINE_string('project', 'test', '')
flags.DEFINE_string('algo_name', 'ask_hilp', '') # 'ask', 'ask_hilp'

flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('batch_size', 1024, '')
flags.DEFINE_integer('pretrain_steps', 500002, '')
flags.DEFINE_integer('eval_interval', 100, '')
flags.DEFINE_integer('save_interval', 100000, '')
flags.DEFINE_integer('log_interval', 1000, '')
flags.DEFINE_integer('eval_episodes', 1, '')
flags.DEFINE_integer('num_video_episodes', 1, '')

flags.DEFINE_integer('way_steps', 25, '')
flags.DEFINE_integer('use_layer_norm', 1, '')
flags.DEFINE_integer('value_hidden_dim', 512, '')
flags.DEFINE_integer('value_num_layers', 3, '')
flags.DEFINE_integer('actor_hidden_dim', 256, '')
flags.DEFINE_integer('actor_num_layers', 2, '')
flags.DEFINE_integer('qf_hidden_dim', 512, '')
flags.DEFINE_integer('qf_num_layers', 3, '')
flags.DEFINE_integer('geom_sample', 1, '')

flags.DEFINE_float('p_randomgoal', 0.1, '') # 0.3
flags.DEFINE_float('p_trajgoal', 0.5, '')
flags.DEFINE_float('p_currgoal', 0.4, '') # 0.2
flags.DEFINE_float('high_p_randomgoal', 0.0, '') # 0.6
flags.DEFINE_float('high_p_relable', 0.8, '')
flags.DEFINE_float('high_temperature', 1, '')
flags.DEFINE_float('pretrain_expectile', 0.7, '')
flags.DEFINE_float('temperature', 1, '')
flags.DEFINE_float('discount', 0.99, '')

flags.DEFINE_float('sparse_data', 0, '') # 100% setting : 0, 30% setting : -7, 10% setting : -9
flags.DEFINE_integer('expert_data_On', 0, '') # 현재 kitchen (reward >= 3), calvin (reward >= 4)만 적용

flags.DEFINE_string('use_rep', '', '') # ["hiql_goal_encoder", "hilp_subgoal_encoder", "hilp_encoder", "vae_encoder"]
flags.DEFINE_integer('rep_normalizing_On', 1, '') # 0: rep_norm 제거 // 1: rep_norm 사용
flags.DEFINE_integer('rep_dim', 10, '')
flags.DEFINE_integer('latent_dim', 10, '')

flags.DEFINE_string('build_keynode_time', "during_training", '') # ["pre_training", "during_training", "post_training"]
flags.DEFINE_integer('keynode_num', 2, '')
flags.DEFINE_integer('kmean_weight_On', 0, '')
flags.DEFINE_integer('use_goal_info_On', 0, '')
flags.DEFINE_string('kmean_weight_type', 'rtg_uniform', '')  # ['rtg_discount', 'rtg_uniform', "hilbert_td"]
flags.DEFINE_integer('specific_dim_On', 1, '')
flags.DEFINE_float('keynode_ratio', 1, '')
flags.DEFINE_integer('use_keynode_in_eval_On', 1, '')

flags.DEFINE_integer('relative_dist_in_eval_On', 0, '')
flags.DEFINE_string('mapping_method', 'nearest', '') # nearest, triple, center

flags.DEFINE_integer('hilp_skill_dim', 32, '')
flags.DEFINE_integer('hilp_decoder', 1, '')
flags.DEFINE_integer('cql_high_target_action_gap', 32, '')
flags.DEFINE_integer('bilinear', 0, '')

flags.DEFINE_integer('vae_encoder_dim', 10, '')
flags.DEFINE_float('vae_recon_coe', 0.00, '')
flags.DEFINE_float('vae_kl_coe', 0.0, '')

# 0610 승호수정 goal only
flags.DEFINE_string('rep_type', 'concat', '') # 'state' / 'concat'
# 0610 승호수정 spherical
flags.DEFINE_float('spherical_On', 0, '') # 0:Euclidean Distance // 1: Cosine Similarity
flags.DEFINE_float('mapping_threshold', 0.0, '')
flags.DEFINE_integer('kl_loss', 0, '')
flags.DEFINE_integer('final_goal', 0, '')
flags.DEFINE_float('mse_loss', 0, '') # train high action to key node (mse loss)
flags.DEFINE_integer('high_action_in_hilp', 0, '') # high action - 0: raw obs 1: hilp space 
flags.DEFINE_integer('low_actor_train_with_high_actor', 0, '') # low actor train with high action - 0: offline obs low goals 1: trained high level policy action 
flags.DEFINE_integer('correction_value', 0, '') # value ratio correction
flags.DEFINE_integer('n_step_hilp', 0, '') # value ratio correction
flags.DEFINE_string('distance', '', '') # value ratio correction

flags.DEFINE_integer('key_node_q', 0, '') # high qf train with key node q
flags.DEFINE_integer('key_node_train', 0, '') # high qf train with key node q

wandb_config = default_wandb_config()
wandb_config.update({
    'project': f'ASK_{project}',
    'group': 'Debug',
    'name': '{env_name}',
})
    
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', learner.get_default_config(), lock_config=False)
gcdataset_config = GCSDataset.get_default_config()
config_flags.DEFINE_config_dict('gcdataset', gcdataset_config, lock_config=False)

@jax.jit
def get_debug_statistics(agent, batch):
    def get_info(s, g):
        return agent.network(s, g, info=True, method='value')
    
    s = batch['observations']
    g = batch['goals']
    
    # if agent.config['use_rep'] == "vae_encoder":
    #     s, g = (agent.network(x, method='vae_state_encoder')[0] for x in [s, g])
    
    # elif agent.config['use_rep'] == "hilp_encoder":
    #     s, g = (agent.network(x, method='hilp_phi') for x in [s, g])
        
    # elif agent.config['use_rep'] =="hilp_subgoal_encoder":
    #     g = (agent.network(g, method='hilp_phi'))
    
    info = get_info(s, g)
    stats = {}
    stats.update({
        'v': info['v'].mean(),
    })
    return stats

@jax.jit
def get_gcvalue(agent, s, g):
    v1, v2 = agent.network(s, g, method='value')
    return (v1 + v2) / 2

def get_v(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    return get_gcvalue(agent, observations, goal)

@jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        v1, v2 = agent.network(jax.tree_map(lambda x: x[None], s), jax.tree_map(lambda x: x[None], g), method='value')
        return (v1 + v2) / 2
    
    observations = trajectory['observations']
    
    if agent.config['use_rep'] == "hiql_goal_encoder":
        all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
        
    elif agent.config['use_rep'] == "vae_encoder":
        observations = agent.network(observations, method='vae_state_encoder')[0]
        all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
        
    elif agent.config['use_rep'] == "hilp_encoder":
        rep_observations = agent.network(observations, method='hilp_phi')        
        all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(rep_observations, rep_observations)
    
    elif agent.config['use_rep'] =="hilp_subgoal_encoder":
        rep_observations = agent.network(observations, method='hilp_phi')    
        all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, rep_observations)
    
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }

def main(_):
    g_start_time = time.strftime('%m-%d_%H-%M')

    exp_name = ''
    exp_name += f'{FLAGS.wandb["name"]}'
    exp_name += f'_sd{FLAGS.seed:03d}'
    exp_name += f'_{g_start_time}'
    
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'

    
    if 'visual' in FLAGS.env_name or 'topview' in FLAGS.env_name :
        # FLAGS.visual = 1
        # FLAGS.p_aug = 0.5
        FLAGS.high_p_randomgoal = 0.0
        FLAGS.batch_size = 256
        FLAGS.actor_hidden_dim = 512
        FLAGS.actor_num_layers = 3
        # assert FLAGS.use_rep != ''
    else:
        FLAGS.actor_hidden_dim = 256
        FLAGS.actor_num_layers = 2
        
    if FLAGS.bilinear != 0:
        FLAGS.qf_hidden_dim = 256
        FLAGS.qf_num_layers = 3
        
        
    FLAGS.gcdataset['p_randomgoal'] = FLAGS.p_randomgoal
    FLAGS.gcdataset['p_trajgoal'] = FLAGS.p_trajgoal
    FLAGS.gcdataset['p_currgoal'] = FLAGS.p_currgoal
    FLAGS.gcdataset['high_p_randomgoal'] = FLAGS.high_p_randomgoal
    FLAGS.gcdataset['high_p_relable'] = FLAGS.high_p_relable
    FLAGS.gcdataset['geom_sample'] = FLAGS.geom_sample
    FLAGS.gcdataset['discount'] = FLAGS.discount
    FLAGS.gcdataset['way_steps'] = FLAGS.way_steps
    FLAGS.gcdataset['final_goal'] = FLAGS.final_goal
    FLAGS.gcdataset['high_action_in_hilp'] = FLAGS.high_action_in_hilp
    
    FLAGS.gcdataset['keynode_ratio'] = FLAGS.keynode_ratio
  
    FLAGS.config['env_name'] = FLAGS.env_name
    FLAGS.config['pretrain_expectile'] = FLAGS.pretrain_expectile
    FLAGS.config['high_temperature'] = FLAGS.high_temperature
    FLAGS.config['temperature'] = FLAGS.temperature
    FLAGS.config['discount'] = FLAGS.discount
    FLAGS.config['way_steps'] = FLAGS.way_steps
    FLAGS.config['value_hidden_dims'] = (FLAGS.value_hidden_dim,) * FLAGS.value_num_layers
    FLAGS.config['actor_hidden_dims'] = (FLAGS.actor_hidden_dim,) * FLAGS.actor_num_layers
    FLAGS.config['qf_hidden_dims'] = (FLAGS.qf_hidden_dim,) * FLAGS.qf_num_layers
    
    FLAGS.config['sparse_data'] = FLAGS.sparse_data
    FLAGS.config['build_keynode_time'] = FLAGS.build_keynode_time
    FLAGS.config['keynode_num'] = FLAGS.keynode_num
    FLAGS.config['kmean_weight_On'] = FLAGS.kmean_weight_On
    FLAGS.config['keynode_ratio'] = FLAGS.keynode_ratio
    FLAGS.config['use_keynode_in_eval_On'] = FLAGS.use_keynode_in_eval_On
    FLAGS.config['use_rep'] = FLAGS.use_rep
    FLAGS.config['hilp_skill_dim'] = FLAGS.hilp_skill_dim 
    FLAGS.config['hilp_decoder'] = FLAGS.hilp_decoder 
    FLAGS.config['vae_recon_coe'] = FLAGS.vae_recon_coe
    FLAGS.config['kl_loss'] = FLAGS.kl_loss
    FLAGS.config['final_goal'] = FLAGS.final_goal
    FLAGS.config['mse_loss'] = FLAGS.mse_loss
    FLAGS.config['high_action_in_hilp'] = FLAGS.high_action_in_hilp
    FLAGS.config['low_actor_train_with_high_actor'] = FLAGS.low_actor_train_with_high_actor
    FLAGS.config['correction_value'] = FLAGS.correction_value
    FLAGS.config['n_step_hilp'] = FLAGS.n_step_hilp
    FLAGS.config['distance'] = FLAGS.distance
    FLAGS.config['key_node_q'] = FLAGS.key_node_q
    FLAGS.config['key_node_train'] = FLAGS.key_node_train
    FLAGS.config['bilinear'] = FLAGS.bilinear
    FLAGS.config['cql_high_target_action_gap'] = FLAGS.cql_high_target_action_gap

    # Create wandb logger
    params_dict = {**FLAGS.gcdataset.to_dict(), **FLAGS.config.to_dict()}
    FLAGS.wandb['name'] = FLAGS.wandb['exp_descriptor'] = exp_name   
    FLAGS.wandb['group'] = FLAGS.wandb['exp_prefix'] = FLAGS.run_group 

    setup_wandb(params_dict, **FLAGS.wandb)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb['group'], FLAGS.env_name + "_" + FLAGS.wandb['name'].split('_', 2)[2])
    os.makedirs(FLAGS.save_dir, exist_ok=True)
        
    # 명령어 로깅
    log_file_name = f"log_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.txt"
    log_file_path = os.path.join(FLAGS.save_dir, log_file_name)
    with open(log_file_path, 'w') as log_file:
        log_file.write("Flags used:\n")
        for flag_name in FLAGS:
            flag_value = getattr(FLAGS, flag_name)
            log_file.write(f"{flag_name}: {flag_value}\n")
    print(f"Log file created at {log_file_path}")


    
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    env, dataset, episode_index, goal_info, dataset_config = d4rl_utils.make_env_get_dataset(FLAGS)

    # env_name = FLAGS.env_name
    # if 'antmaze' in FLAGS.env_name:
    #     if 'ultra' in FLAGS.env_name:
    #         import d4rl_ext
    #         import gym
    #         env = gym.make(env_name)
    #         env = EpisodeMonitor(env)
    #         env.seed(FLAGS.seed)
    #     else:
    #         env = d4rl_utils.make_env(env_name)
    #         env.seed(FLAGS.seed)   
    #     dataset, dataset_config = d4rl_utils.get_dataset(env, FLAGS.env_name, flag=FLAGS)
    #     dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})
    #     env.render(mode='rgb_array', width=500, height=500)
    #     if 'large' in FLAGS.env_name:
    #         env.viewer.cam.lookat[0] = 18
    #         env.viewer.cam.lookat[1] = 12
    #         env.viewer.cam.distance = 50
    #         env.viewer.cam.elevation = -90
    #         viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
    #         viz = ant_diagnostics.Visualizer(env_name, viz_env, viz_dataset, discount=FLAGS.discount)
    #         init_state = np.copy(viz_dataset['observations'][0])
    #         init_state[:2] = (12.5, 8)
    #     elif 'ultra' in FLAGS.env_name:
    #         env.viewer.cam.lookat[0] = 26
    #         env.viewer.cam.lookat[1] = 18
    #         env.viewer.cam.distance = 70
    #         env.viewer.cam.elevation = -90
    #     else:
    #         env.viewer.cam.lookat[0] = 18
    #         env.viewer.cam.lookat[1] = 12
    #         env.viewer.cam.distance = 50
    #         env.viewer.cam.elevation = -90
    # elif 'kitchen' in FLAGS.env_name:
    #     env = d4rl_utils.make_env(FLAGS.env_name)
    #     env.seed(FLAGS.seed)
    #     dataset, dataset_config = d4rl_utils.get_dataset(env, FLAGS.env_name, flag=FLAGS, filter_terminals=True)
    #     dataset = dataset.copy({'observations': dataset['observations'][:, :30], 'next_observations': dataset['next_observations'][:, :30]})
    # elif 'calvin' in FLAGS.env_name:
    #     from src.envs.calvin import CalvinEnv
    #     from hydra import compose, initialize
    #     from src.envs.gym_env import GymWrapper
    #     from src.envs.gym_env import wrap_env
    #     initialize(config_path='src/envs/conf')
    #     cfg = compose(config_name='calvin')
    #     env = CalvinEnv(**cfg)
    #     env.seed(FLAGS.seed)
    #     env.max_episode_steps = cfg.max_episode_steps = 360
    #     env = GymWrapper(
    #         env=env,
    #         from_pixels=cfg.pixel_ob,
    #         from_state=cfg.state_ob,
    #         height=cfg.screen_size[0],
    #         width=cfg.screen_size[1],
    #         channels_first=False,
    #         frame_skip=cfg.action_repeat,
    #         return_state=False,
    #     )
    #     env = wrap_env(env, cfg)
    #     data = pickle.load(gzip.open(os.path.dirname(os.path.realpath(__file__)) + '/data/calvin.gz', "rb")) # 현재 실행되는 파일 위치에서 calvin 파일 찾음
    #     ds = []
    #     episode_index = 0
    #     for i, d in enumerate(data):
    #         if len(d['obs']) < len(d['dones']):
    #             continue  # Skip incomplete trajectories.
    #         # Only use the first 21 states of non-floating objects.
    #         d['obs'] = d['obs'][:, :21]
    #         new_d = dict(
    #             observations=d['obs'][:-1],
    #             next_observations=d['obs'][1:],
    #             actions=d['actions'][:-1],
    #             episodes = [episode_index]*len(d['obs'][:-1])
    #         )
    #         num_steps = new_d['observations'].shape[0]
    #         new_d['rewards'] = np.zeros(num_steps)
    #         new_d['terminals'] = np.zeros(num_steps, dtype=bool)
    #         new_d['terminals'][-1] = True
    #         ds.append(new_d)
    #         episode_index +=1
    #     dataset = dict()
    #     for key in ds[0].keys():
    #         dataset[key] = np.concatenate([d[key] for d in ds], axis=0)
    #     dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, dataset=dataset, flag=FLAGS)
    # else:
    #     raise NotImplementedError

    total_steps = FLAGS.pretrain_steps
    example_observation = dataset['observations'][0, np.newaxis]
    example_action = dataset['actions'][0, np.newaxis]

    
    if 'ask' == FLAGS.algo_name:
        from src.agents import ask as learner
    elif 'ask_hilp' == FLAGS.algo_name:
        from src.agents import ask_hilp as learner
    elif 'cql' == FLAGS.algo_name:
        from src.agents import cql as learner
        # target_entropy = -np.prod(env.action_space.shape).item()
        # high_target_entropy = -np.prod(env.observation_space.shape).item()
        # cql_parameters = {'target_entropy':target_entropy, 'high_target_entropy':high_target_entropy}
        # FLAGS.config.update(cql_parameters)
        FLAGS.config.update(dataset_config)
    elif 'guider' == FLAGS.algo_name:
        from src.agents import guider as learner
        FLAGS.config.update(dataset_config)

        
    
    
    
    
    agent = learner.create_learner(FLAGS.seed,
                                   example_observation,
                                   example_action,
                                   use_layer_norm=FLAGS.use_layer_norm,
                                   flag=FLAGS,
                                   
                                   **FLAGS.config)
    
    
    # hilp 이외의 알고리즘 초기화 용
    find_key_node, hilp_fn, distance_fn, key_nodes, decode = None, None, None, None, None
    
    if 'guider' == FLAGS.algo_name:
        decode = agent.get_decode
    
    # if FLAGS.use_rep == "hiql_goal_encoder":
    #     encoder_fn = jax.jit(jax.vmap(agent.get_value_goal))
    #     # 0610 승호수정 spherical
    #     if FLAGS.rep_type == 'concat':
    #         rep_observations = d4rl_utils.get_rep_observation_spherical(encoder_fn, dataset, FLAGS)
    #     elif FLAGS.rep_type == 'state':
    #         rep_observations = d4rl_utils.get_rep_observation_goal_only(encoder_fn, dataset, FLAGS)
    #     dataset = d4rl_utils.add_data(dataset, rep_observations)
        
    # elif FLAGS.use_rep == "vae_encoder":
    #     encoder_fn = jax.jit(jax.vmap(agent.get_vae_state_rep))
    #     rep_observations = d4rl_utils.get_rep_observation(encoder_fn, dataset, FLAGS)
    #     dataset = d4rl_utils.add_data(dataset, rep_observations)
            
    # elif FLAGS.use_rep in ["hilp_subgoal_encoder", "hilp_encoder"]:
    #     encoder_fn = jax.jit(jax.vmap(agent.get_hilp_phi))
    #     rep_observations = d4rl_utils.get_hilp_rep_observation(encoder_fn, dataset, FLAGS)
    #     if FLAGS.kmean_weight_type == 'hilbert_td':
    #         dataset = d4rl_utils.hilp_add_data(dataset, rep_observations)
    #     elif FLAGS.kmean_weight_type in ['rtg_discount', 'rtg_uniform']:
    #         dataset = d4rl_utils.add_data(dataset, rep_observations)
            
        
    # if FLAGS.config['build_keynode_time'] in ["pre_training", "during_training"]:       
    #     key_nodes, sparse_data_index = keynode_utils.build_keynodes(dataset, flags=FLAGS, episode_index= episode_index)
    #     if FLAGS.use_rep in ["hiql_goal_encoder", "vae_encoder", "hilp_subgoal_encoder", "hilp_encoder"]:
    #         # 0610 승호수정 spherical
    #         key_nodes.construct_nodes(rep_observations=rep_observations, spherical_On=FLAGS.spherical_On)
    # if FLAGS.use_keynode_in_eval_On:
    #     find_key_node = jax.jit(key_nodes.find_closest_node)
    #     # agent = agent.replace(key_nodes = key_nodes.pos)
    # else:
    #     find_key_node = None
        
    # if FLAGS.sparse_data:
    #     dataset = d4rl_utils.sparse_data(dataset, sparse_data_index=sparse_data_index)





    pretrain_dataset = GCSDataset(dataset, **FLAGS.gcdataset.to_dict())

        
    encoder_fn = None
    decoder_fn = None
    value_goal_fn = None
        
    # For debugging metrics
    if 'antmaze' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(1000, 1050))
    elif 'kitchen' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    elif 'calvin' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    else:
        raise NotImplementedError
    
    FLAGS.config['obs_dim'] = example_observation[0].shape[-1]

    base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
    observation = env.reset()
    
    if 'antmaze' in FLAGS.env_name:
        goal = env.wrapped_env.target_goal
        obs_goal = base_observation.copy()
        obs_goal[:2] = goal
    elif 'kitchen' in FLAGS.env_name:
        if 'visual' not in FLAGS.env_name:
            observation, obs_goal = observation[:30].copy(), observation[30:].copy()
            obs_goal[:9] = base_observation[:9]

    elif 'calvin' in FLAGS.env_name:
        observation = observation['ob']
        goal = np.array([0.25, 0.15, 0, 0.088, 1, 1])
        obs_goal = base_observation.copy()
        obs_goal[15:21] = goal
    
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    return_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'return.csv'))
    first_time = time.time()
    last_time = time.time()
    

    if False:
        if 'antmaze' in FLAGS.env_name:
            load_file = '/home/qortmdgh4141/disk/HIQL_Team_Project/TG/data/ant_hilp_64.pkl'
        elif 'visual-kitchen' in FLAGS.env_name:
            load_file = '/home/qortmdgh4141/disk/HIQL_Team_Project/TG/data/visual_kitchen_miexd_hilp_128.pkl'
        elif 'kitchen' in FLAGS.env_name:
            load_file = '/home/qortmdgh4141/disk/HIQL_Team_Project/TG/data/kitchen_mixed_hilp.pkl'
        elif 'calvin' in FLAGS.env_name:
            load_file = '/home/qortmdgh4141/disk/HIQL_Team_Project/TG/data/kitchen_mixed_hilp.pkl'
        else:
            NotImplementedError
    else:
        load_file = None
    
    # pre train - hilp or prior model
    if 'ask' in FLAGS.algo_name or 'cql' in FLAGS.algo_name or 'guider' in FLAGS.algo_name:
        if load_file is None:
            
            # train_steps = int(2*10**5 + 1) if 'ant' in FLAGS.env_name else int(1*10**5 + 1)
            pretrain_steps = int(5*10**3 + 1) if 'guider' in FLAGS.algo_name else pretrain_steps
            if 'guider' in FLAGS.algo_name:
                update = dict(qf_update=False, actor_update=False, alpha_update=False, high_actor_update=False, high_qf_update=False, hilp_update=False, prior_update=True)
            else:
                update = dict(qf_update=False, actor_update=False, alpha_update=False, high_actor_update=False, high_qf_update=False, hilp_update=True, prior_update=False)
            
            
            for i in tqdm.tqdm(range(1, pretrain_steps),
                        desc="pre_train",
                        smoothing=0.1,
                        dynamic_ncols=True):
                
                pretrain_batch = pretrain_dataset.sample(FLAGS.batch_size, **update)
                agent, update_info = supply_rng(agent.pretrain_update)(pretrain_batch, **update)
                if i % FLAGS.log_interval == 0:
                    train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                
                    if i % (FLAGS.log_interval *10) == 0:
                        pretrain_batch = pretrain_dataset.sample(FLAGS.batch_size)
                        if 'ant' in FLAGS.env_name and 'networks_hilp_value' in agent.network.params.keys():
                            value_map, identity_map = plot_value_map(agent, base_observation, obs_goal, i, g_start_time, pretrain_batch, dataset['observations'])
                            train_metrics['value_map'] = wandb.Image(value_map)
                        # else: 
                        # ant 외 환경에서 tsne 사용해서 value 그려볼것
                        #     value_map, identity_map = plot_value_map_others(agent, base_observation, obs_goal, i, g_start_time, pretrain_batch, dataset['observations'])
                            
                    wandb.log(train_metrics, step=i)
            
            save_dict = dict(
            agent=flax.serialization.to_state_dict(agent),
            config=FLAGS.config.to_dict())

            fname = os.path.join(FLAGS.save_dir, f'params_hilp.pkl')
            print(f'Saving to {fname}')
            
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)  
        else: 
            i=0
            train_metrics= dict()
            with open(load_file, "rb") as f:
                loaded_dict = pickle.load(f)
            agent = flax.serialization.from_state_dict(agent, loaded_dict['agent'])
        
        if 'networks_hilp_value' in agent.network.params.keys():
            distance_fn = jax.jit(agent.get_hilp_value)
            filtered_transition_index, hlip_filtered_index, dones_indexes = d4rl_utils.get_transition_index(agent, dataset, FLAGS)
            
            hilp_observations = d4rl_utils.get_hilp_obs(agent, dataset['observations'], FLAGS)
            dataset = d4rl_utils.add_data(dataset, rep_observations=hilp_observations)
            
            key_nodes, sparse_data_index = keynode_utils.build_keynodes(dataset, flags=FLAGS, episode_index=filtered_transition_index, rep_obs=hilp_observations)
            agent = agent.replace(key_nodes=key_nodes.centroids)
            hilp_fn = jax.jit(agent.get_hilp_phi)
        elif FLAGS.use_keynode_in_eval_On:
            key_nodes, sparse_data_index = keynode_utils.build_keynodes(dataset, flags=FLAGS)
            agent = agent.replace(key_nodes=key_nodes.centroids)
            
        
        if FLAGS.high_action_in_hilp:
            dataset = dataset.copy({'rep_observations' : hilp_observations})
            agent = agent.replace(config=agent.config.copy({'rep_observation_min':jnp.min(hilp_observations, axis=0), 'rep_observation_max':jnp.max(hilp_observations, axis=0), }))

        elif 'guider' not in FLAGS.algo_name:
            dataset = dataset.copy({'key_node' : key_nodes.matched_keynode_in_raw})
            dataset = dataset.copy({'rep_observations' : hilp_observations})
            
        pretrain_dataset = GCSDataset(dataset, **FLAGS.gcdataset.to_dict())

        if i >= FLAGS.log_interval:
            if 'ant' in FLAGS.env_name and 'networks_hilp_value' in agent.network.params.keys():
                transition_index = (filtered_transition_index, hlip_filtered_index, dones_indexes)
                pretrain_batch = pretrain_dataset.sample(FLAGS.batch_size)
                value_map, identity_map = plot_value_map(agent, base_observation, obs_goal, 0, g_start_time, pretrain_batch, dataset['observations'], transition_index, key_node=key_nodes)
                train_metrics['key_node_map'] = wandb.Image(value_map)
            
            # else:
            #     value_map, identity_map = plot_value_map_others(agent, base_observation, obs_goal, 0, g_start_time, pretrain_batch, dataset['observations'], transition_index, key_node=key_nodes)
                
            wandb.log(train_metrics, step=i)
            
            
    
    update = dict(qf_update=False, actor_update=False, alpha_update=True, high_actor_update=True, high_qf_update=True, hilp_update=False, prior_update=False)
    
    for i in tqdm.tqdm(range(1, total_steps + 1),
                   desc="main_train",
                   smoothing=0.1,
                   dynamic_ncols=True):
        if i == 5e5:
            update = dict(qf_update=True, actor_update=True, alpha_update=False, high_actor_update=False, high_qf_update=False, hilp_update=False, prior_update=False)
            
        pretrain_batch = pretrain_dataset.sample(FLAGS.batch_size, **update)
        # agent, update_info = supply_rng(agent.pretrain_update)(pretrain_batch, hilp_update=False)
        agent, update_info = supply_rng(agent.pretrain_update)(pretrain_batch, **update)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if FLAGS.algo_name not in ['cql', 'guider']:
                debug_statistics = get_debug_statistics(agent, pretrain_batch)
                train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            last_time = time.time()
            wandb.log(train_metrics, step=pretrain_steps+i)
            train_logger.log(train_metrics, step=pretrain_steps+i)


        if i == 1 or i % FLAGS.eval_interval == 0:

            eval_episodes = 1 if i == 1 else FLAGS.eval_episodes
            num_video_episodes = 0 if i == 1 else FLAGS.num_video_episodes
            
            policy_fn = partial(supply_rng(agent.sample_actions))
            high_policy_fn = partial(supply_rng(agent.sample_high_actions))
            base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])

            eval_info, trajs, renders, rep_trajectories, cos_distances = evaluate_with_trajectories(
                    policy_fn=policy_fn, high_policy_fn=high_policy_fn, env=env,
                    env_name=FLAGS.env_name, num_episodes=eval_episodes,
                    base_observation=base_observation, num_video_episodes=num_video_episodes,
                    eval_temperature=0,
                    config=FLAGS.config,
                    find_key_node=find_key_node,
                    hilp_fn=hilp_fn,
                    distance_fn=distance_fn,
                    FLAGS=FLAGS,
                    nodes=key_nodes,
                    decode=decode,
                )
            
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            if FLAGS.num_video_episodes > 0 and len(renders):
                video = record_video('Video', i, renders=renders)
                eval_metrics['video'] = video
            print(f"{eval_info['evaluation/episode/return']*100=}")
            if 'ant' in FLAGS.env_name and 'ask' == FLAGS.algo_name:
                value_map, identity_map = plot_value_map(agent, base_observation, obs_goal, i, g_start_time, pretrain_batch, dataset['observations'], transition_index)
                eval_metrics['value_map'] = wandb.Image(value_map)
                
            elif 'ant' in FLAGS.env_name and 'cql' in FLAGS.algo_name:
                if FLAGS.high_action_in_hilp:
                    q_map = plot_q_map_modified(agent, base_observation, obs_goal, i, g_start_time, pretrain_batch, dataset['observations'], trajs=trajs)
                else:
                    q_map = plot_q_map(agent, base_observation, obs_goal, i, g_start_time, pretrain_batch, dataset['observations'], trajs=trajs)
                eval_metrics['q_map'] = wandb.Image(q_map)
                
            # traj_metrics = get_traj_v(agent, example_trajectory)
            # value_viz = viz_utils.make_visual_no_image(
            #     traj_metrics,
            #     [partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
            # )
            # eval_metrics['value_traj_viz'] = wandb.Image(value_viz)
            # if 'antmaze' in FLAGS.env_name and 'large' in FLAGS.env_name and FLAGS.env_name.startswith('antmaze'):
            #     traj_image = d4rl_ant.trajectory_image(viz_env, viz_dataset, trajs)
                # eval_metrics['trajectories'] = wandb.Image(traj_image)
                # new_metrics_dist = viz.get_distance_metrics(trajs)
                # eval_metrics.update({
                    # f'debugging/{k}': v for k, v in new_metrics_dist.items()})
                # image_v = d4rl_ant.gcvalue_image(
                #     viz_env,
                #     viz_dataset,
                #     partial(get_v, agent),
                # )
                # eval_metrics['v'] = wandb.Image(image_v)

            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
                config=FLAGS.config.to_dict()
            )
            if i == 1 or i % FLAGS.save_interval == 0:
                # key_node_fname = os.path.join(FLAGS.save_dir, f'key_node_{i}.pkl')
                # with open(key_node_fname, 'wb') as f:
                #     pickle.dump(np.array(key_nodes.centroids), f)
                
                # cos_distances_path = os.path.join(FLAGS.save_dir, f'cos_distances_{i}.pkl')
                # with open(cos_distances_path, 'wb') as f:
                #     pickle.dump(np.array(cos_distances), f)
                
                fname = os.path.join(FLAGS.save_dir, f'params_{i}.pkl')
                print(f'Saving to {fname}')
                with open(fname, "wb") as f:
                    pickle.dump(save_dict, f)         
            if 'calvin' in FLAGS.env_name:
                score = eval_metrics['evaluation/final.return']
                log_key = 'evaluation/final.return'
            else:
                score = eval_metrics['evaluation/episode.return']
                log_key = 'evaluation/episode.return'
            wandb.log(eval_metrics, step=pretrain_steps+i)
            eval_logger.log(eval_metrics, step=pretrain_steps+i)
            
            return_logger.log({'evaluation/episode.return' :eval_metrics[log_key]}, step=pretrain_steps+i)
    train_logger.close()
    eval_logger.close()
    return_logger.close()

if __name__ == '__main__':
    import random
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    
    app.run(main)
