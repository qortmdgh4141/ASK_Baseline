import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
tf.config.optimizer.set_jit(True) 

from absl import app, flags
from functools import partial
from src.agents import ask as learner
from src.gc_dataset import GCSDataset
from ml_collections import config_flags
from src.utils import record_video, CsvLogger, plot_value_map
from jaxrl_m.wandb import setup_wandb, default_wandb_config
from src import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils, keynode_utils
from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories, EpisodeMonitor

FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', f'/home/spectrum/study/experiment_output/', '')
flags.DEFINE_string('run_group', 'EXP', '')
flags.DEFINE_string('env_name', 'FetchPush-v1-mixed', '') # 'FetchPush-v1
# flags.DEFINE_string('env_name', 'antmaze-ultra-diverse-v0', '')
flags.DEFINE_string('project', 'test', '')
flags.DEFINE_string('algo_name', None, '')

flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('batch_size', 1024, '')
flags.DEFINE_integer('pretrain_steps', 500002, '')
flags.DEFINE_integer('eval_interval', 1000, '')
flags.DEFINE_integer('save_interval', 100000, '')
flags.DEFINE_integer('log_interval', 1000, '')
flags.DEFINE_integer('eval_episodes', 1, '')
flags.DEFINE_integer('num_video_episodes', 2, '')

flags.DEFINE_integer('way_steps', 35, '')
flags.DEFINE_integer('use_layer_norm', 1, '')
flags.DEFINE_integer('value_hidden_dim', 512, '')
flags.DEFINE_integer('value_num_layers', 3, '')
flags.DEFINE_integer('actor_hidden_dim', 256, '')
flags.DEFINE_integer('actor_num_layers', 3, '')
flags.DEFINE_integer('geom_sample', 1, '')

flags.DEFINE_float('p_randomgoal', 0.3, '')
flags.DEFINE_float('p_trajgoal', 0.5, '')
flags.DEFINE_float('p_currgoal', 0.2, '')
flags.DEFINE_float('high_p_randomgoal', 0.3, '')
flags.DEFINE_float('high_temperature', 1, '')
flags.DEFINE_float('pretrain_expectile', 0.7, '')
flags.DEFINE_float('temperature', 1, '')
flags.DEFINE_float('discount', 0.99, '')
flags.DEFINE_integer('visual', 0, '')
flags.DEFINE_float('p_aug', 0, '')

flags.DEFINE_float('sparse_data', 0, '') # 100% setting : 0, 30% setting : -7, 10% setting : -9
flags.DEFINE_integer('expert_data_On', 0, '') # 현재 kitchen (reward >= 3), calvin (reward >= 4)만 적용

flags.DEFINE_string('use_rep', '0', '') # ["hiql_goal_encoder", "hilp_subgoal_encoder", "hilp_encoder", "vae_encoder"]
flags.DEFINE_integer('rep_normalizing_On', 1, '') # 0: rep_norm 제거 // 1: rep_norm 사용
flags.DEFINE_integer('rep_dim', 10, '')
flags.DEFINE_integer('keynode_dim', 10, '')

flags.DEFINE_string('build_keynode_time', "during_training", '') # ["pre_training", "during_training", "post_training"]
flags.DEFINE_integer('keynode_num', 300, '')
flags.DEFINE_integer('kmean_weight_On', 1, '')
flags.DEFINE_integer('use_goal_info_On', 0, '')
flags.DEFINE_string('kmean_weight_type', 'rtg_uniform', '')  # ['rtg_discount', 'rtg_uniform', "hilbert_td"]
flags.DEFINE_integer('specific_dim_On', 0, '')
flags.DEFINE_float('keynode_ratio', 0.0, '')
flags.DEFINE_integer('use_keynode_in_eval_On', 1, '')

flags.DEFINE_integer('relative_dist_in_eval_On', 0, '')
flags.DEFINE_string('mapping_method', 'nearest', '') # nearest, triple, center

flags.DEFINE_integer('hilp_skill_dim', 0, '')

flags.DEFINE_integer('vae_encoder_dim', 10, '')
flags.DEFINE_float('vae_recon_coe', 0.00, '')
flags.DEFINE_float('vae_kl_coe', 0.0, '')

# 0610 승호수정 goal only
flags.DEFINE_string('rep_type', 'state', '') # 'state' / 'concat'
# 0610 승호수정 spherical
flags.DEFINE_float('spherical_On', 0, '') # 0:Euclidean Distance // 1: Cosine Similarity
flags.DEFINE_float('mapping_threshold', 0, '')

flags.DEFINE_string('value_function_num', 'flat', '') # 'flat' / 'hierarchy'
flags.DEFINE_string('low_dim_clustering', '', '') #[tsne_dim, pca_dim, ''] ex) hilp_2 : hilp - 2dim, pca_2 : pca - 2dim
flags.DEFINE_float('pseudo_obs', 0, '') #


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
    
    if agent.config['use_rep'] == "vae_encoder":
        s, g = (agent.network(x, method='vae_state_encoder')[0] for x in [s, g])
    
    elif agent.config['use_rep'] == "hilp_encoder":
        s, g = (agent.network(x, method='hilp_phi') for x in [s, g])
        
    elif agent.config['use_rep'] =="hilp_subgoal_encoder":
        g = (agent.network(g, method='hilp_phi'))
    
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
    
    else:
        all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(
        0, None))(observations, observations)
    
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
        
    if 'visual' in FLAGS.env_name:
        FLAGS.visual = 1
        FLAGS.p_aug = 0.5
        
    if FLAGS.visual:
        FLAGS.batch_size = 256

    FLAGS.gcdataset['p_randomgoal'] = FLAGS.p_randomgoal
    FLAGS.gcdataset['p_trajgoal'] = FLAGS.p_trajgoal
    FLAGS.gcdataset['p_currgoal'] = FLAGS.p_currgoal
    FLAGS.gcdataset['high_p_randomgoal'] = FLAGS.high_p_randomgoal
    FLAGS.gcdataset['geom_sample'] = FLAGS.geom_sample
    FLAGS.gcdataset['discount'] = FLAGS.discount
    FLAGS.gcdataset['way_steps'] = FLAGS.way_steps
    
    FLAGS.gcdataset['keynode_ratio'] = FLAGS.keynode_ratio
    FLAGS.gcdataset['hierarchy'] = FLAGS.value_function_num
    FLAGS.gcdataset['pseudo_obs'] = FLAGS.pseudo_obs
    FLAGS.gcdataset['env_name'] = FLAGS.env_name
  
    FLAGS.config['env_name'] = FLAGS.env_name
    FLAGS.config['pretrain_expectile'] = FLAGS.pretrain_expectile
    FLAGS.config['high_temperature'] = FLAGS.high_temperature
    FLAGS.config['temperature'] = FLAGS.temperature
    FLAGS.config['discount'] = FLAGS.discount
    FLAGS.config['way_steps'] = FLAGS.way_steps
    FLAGS.config['value_hidden_dims'] = (FLAGS.value_hidden_dim,) * FLAGS.value_num_layers
    if FLAGS.visual:
        FLAGS.actor_hidden_dim = 512
        FLAGS.actor_num_layers = 3
        assert FLAGS.use_rep != '0'
    else:
        FLAGS.actor_hidden_dim = 256
        FLAGS.actor_num_layers = 2
    FLAGS.config['actor_hidden_dims'] = (FLAGS.actor_hidden_dim,) * FLAGS.actor_num_layers
    
    FLAGS.config['sparse_data'] = FLAGS.sparse_data
    FLAGS.config['build_keynode_time'] = FLAGS.build_keynode_time
    FLAGS.config['keynode_num'] = FLAGS.keynode_num
    FLAGS.config['kmean_weight_On'] = FLAGS.kmean_weight_On
    FLAGS.config['keynode_ratio'] = FLAGS.keynode_ratio
    FLAGS.config['use_keynode_in_eval_On'] = FLAGS.use_keynode_in_eval_On
    FLAGS.config['use_rep'] = FLAGS.use_rep
    FLAGS.config['hilp_skill_dim'] = FLAGS.hilp_skill_dim 
    FLAGS.config['vae_recon_coe'] = FLAGS.vae_recon_coe

    FLAGS.config['value_function_num'] = FLAGS.value_function_num
    FLAGS.config['pseudo_obs'] = FLAGS.pseudo_obs
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

    env_name = FLAGS.env_name
    if 'antmaze' in FLAGS.env_name:
        if 'ultra' in FLAGS.env_name:
            import d4rl_ext
            import gym
            env = gym.make(env_name)
            env = EpisodeMonitor(env)
            env.seed(FLAGS.seed)
        else:
            env = d4rl_utils.make_env(env_name)
            env.seed(FLAGS.seed)   
        dataset, episode_index = d4rl_utils.get_dataset(env, FLAGS.env_name, flag=FLAGS)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})
        env.render(mode='rgb_array', width=500, height=500)
        if 'large' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90
            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            viz = ant_diagnostics.Visualizer(env_name, viz_env, viz_dataset, discount=FLAGS.discount)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)
        elif 'ultra' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 26
            env.viewer.cam.lookat[1] = 18
            env.viewer.cam.distance = 70
            env.viewer.cam.elevation = -90
        else:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90
    elif 'kitchen' in FLAGS.env_name:
        if 'visual' in FLAGS.env_name:
            from src.d4rl_utils import kitchen_render
            orig_env_name = FLAGS.env_name.split('visual-')[1]
            env = d4rl_utils.make_env(orig_env_name)
            cur_folder = os.path.dirname(__file__)
            dataset = dict(np.load(os.path.join(cur_folder, f'data/d4rl_kitchen_rendered_kitchen-mixed-v0.npz'))) 
            dataset, episode_index = d4rl_utils.get_dataset(env, FLAGS.env_name, dataset=dataset, filter_terminals=True, flag=FLAGS)

            state = env.reset()
            # Random example state from the dataset for proprioceptive states
            goal_state = [-2.3403780e+00, -1.3053924e+00, 1.1021180e+00, -1.8613019e+00, 1.5087037e-01, 1.7687809e+00, 1.2525779e+00, 2.9698312e-02, 3.0899283e-02, 3.9908718e-04, 4.9550228e-05, -1.9946630e-05, 2.7519276e-05, 4.8786267e-05, 3.2835731e-05, 2.6504624e-05, 3.8422750e-05, -6.9888681e-01, -5.0150707e-02, 3.4855098e-01, -9.8701166e-03, -7.6958216e-03, -8.0031347e-01, -1.9142720e-01, 7.2064394e-01, 1.6191028e+00, 1.0021452e+00, -3.2998802e-04, 3.7205056e-05, 5.3616576e-02]
            goal_state[9:] = state[39:]  # Set goal object states
            env.sim.set_state(np.concatenate([goal_state, env.init_qvel]))
            env.sim.forward()
            goal_info = {
                'ob': kitchen_render(env).astype(np.float32),
            }
            env.seed(FLAGS.seed)
            env.reset()
        else:
            env = d4rl_utils.make_env(FLAGS.env_name)
            env.seed(FLAGS.seed)
            dataset, episode_index = d4rl_utils.get_dataset(env, FLAGS.env_name, filter_terminals=True)
            dataset = dataset.copy({'observations': dataset['observations'][:, :30], 'next_observations': dataset['next_observations'][:, :30]})
    elif 'calvin' in FLAGS.env_name:
        from src.envs.calvin import CalvinEnv
        from hydra import compose, initialize
        from src.envs.gym_env import GymWrapper
        from src.envs.gym_env import wr4ap_env
        initialize(config_path='src/envs/conf')
        cfg = compose(config_name='calvin')
        env = CalvinEnv(**cfg)
        env.seed(FLAGS.seed)
        env.max_episode_steps = cfg.max_episode_steps = 360
        env = GymWrapper(
            env=env,
            from_pixels=cfg.pixel_ob,
            from_state=cfg.state_ob,
            height=cfg.screen_size[0],
            width=cfg.screen_size[1],
            channels_first=False,
            frame_skip=cfg.action_repeat,
            return_state=False,
        )
        env = wrap_env(env, cfg)
        data = pickle.load(gzip.open(os.path.dirname(os.path.realpath(__file__)) + '/data/calvin.gz', "rb")) # 현재 실행되는 파일 위치에서 calvin 파일 찾음
        ds = []
        episode_index = 0
        for i, d in enumerate(data):
            if len(d['obs']) < len(d['dones']):
                continue  # Skip incomplete trajectories.
            # Only use the first 21 states of non-floating objects.
            d['obs'] = d['obs'][:, :21]
            new_d = dict(
                observations=d['obs'][:-1],
                next_observations=d['obs'][1:],
                actions=d['actions'][:-1],
                episodes = [episode_index]*len(d['obs'][:-1])
            )
            num_steps = new_d['observations'].shape[0]
            new_d['rewards'] = np.zeros(num_steps)
            new_d['terminals'] = np.zeros(num_steps, dtype=bool)
            new_d['terminals'][-1] = True
            ds.append(new_d)
            episode_index +=1
        dataset = dict()
        for key in ds[0].keys():
            dataset[key] = np.concatenate([d[key] for d in ds], axis=0)
        dataset, episode_index = d4rl_utils.get_dataset(env, FLAGS.env_name, dataset=dataset, flags=FLAGS)
    elif 'Fetch' in FLAGS.env_name:
        import gymnasium as gym
        from src.envs.fetch import fetch_load, FetchGoalWrapper
        
        env = gym.make(FLAGS.env_name.split('-')[0], render_mode='rgb_array',  max_episode_steps=50)
        env.reset(seed=FLAGS.seed)
        env = FetchGoalWrapper(env, FLAGS.env_name)
        env = EpisodeMonitor(env)
        
        env_name, version, type_ = FLAGS.env_name.split('-')
        dataset_file = os.path.join(f'/home/spectrum/study/ASK_Baseline/data/{type_}/{env_name}/buffer.pkl')
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            print(f'{dataset_file}, fetch dataset loaded')
        dataset, episode_index = fetch_load(FLAGS.env_name, dataset)
        # dataset, episode_index = d4rl_utils.get_dataset(env, FLAGS.env_name, flag=FLAGS)
        
    else:
        raise NotImplementedError

    total_steps = FLAGS.pretrain_steps
    example_observation = dataset['observations'][0, np.newaxis]
    example_action = dataset['actions'][0, np.newaxis]
    if 'Fetch' in FLAGS.env_name:
        example_goals = dataset['goal_info'][0, np.newaxis]
    else:
        example_goals = example_observation
        

    agent = learner.create_learner(FLAGS.seed,
                                   example_observation,
                                   example_goals,
                                   example_action,
                                #    value_hidden_dims=FLAGS.config['value_hidden_dims'],
                                #    action_hidden_dims=FLAGS.config['actor_hidden_dims'],
                                   use_layer_norm=FLAGS.use_layer_norm,
                                   visual=FLAGS.visual,
                                   flag=FLAGS,
                                   **FLAGS.config)
    
    if FLAGS.visual == 1:
        encoder_fn = jax.jit(jax.vmap(agent.get_value_goal))
        if FLAGS.rep_type == 'concat':
            rep_observations = d4rl_utils.get_rep_observation_spherical_in_visual(encoder_fn, dataset, FLAGS)
        elif FLAGS.rep_type == 'state':
            rep_observations = d4rl_utils.get_rep_observation_goal_only_in_visual(encoder_fn, dataset, FLAGS)
        dataset = d4rl_utils.add_data(dataset, rep_observations=rep_observations)
    
    elif FLAGS.use_rep == "hiql_goal_encoder":
        encoder_fn = jax.jit(jax.vmap(agent.get_value_goal))
        # 0610 승호수정 spherical
        if FLAGS.rep_type == 'concat':
            rep_observations = d4rl_utils.get_rep_observation_spherical(encoder_fn, dataset, FLAGS)
        elif FLAGS.rep_type == 'state':
            rep_observations = d4rl_utils.get_rep_observation_goal_only(encoder_fn, dataset, FLAGS)
        dataset = d4rl_utils.add_data(dataset, rep_observations=rep_observations)
        
    elif FLAGS.use_rep == "vae_encoder":
        encoder_fn = jax.jit(jax.vmap(agent.get_vae_state_rep))
        rep_observations = d4rl_utils.get_rep_observation(encoder_fn, dataset, FLAGS)
        dataset = d4rl_utils.add_data(dataset, rep_observations=rep_observations)
            
    elif FLAGS.use_rep in ["hilp_subgoal_encoder", "hilp_encoder"]:
        encoder_fn = jax.jit(jax.vmap(agent.get_hilp_phi))
        rep_observations = d4rl_utils.get_hilp_rep_observation(encoder_fn, dataset, FLAGS)
        if FLAGS.kmean_weight_type == 'hilbert_td':
            dataset = d4rl_utils.hilp_add_data(dataset, rep_observations=rep_observations)
        elif FLAGS.kmean_weight_type in ['rtg_discount', 'rtg_uniform']:
            dataset = d4rl_utils.add_data(dataset, rep_observations=rep_observations)
            
        
    if FLAGS.config['build_keynode_time'] in ["pre_training", "during_training"]:       
        key_nodes, sparse_data_index = keynode_utils.build_keynodes(dataset, flags=FLAGS, episode_index= episode_index)
        if FLAGS.use_rep in ["hiql_goal_encoder", "vae_encoder", "hilp_subgoal_encoder", "hilp_encoder"]:
            # 0610 승호수정 spherical
            key_nodes.construct_nodes(rep_observations=rep_observations, spherical_On=FLAGS.spherical_On)
        else:
            # 0621 태건 수정 rep 사용하지 않을때 arg 없이 construct node
            key_nodes.construct_nodes() 
            
        find_key_node = jax.jit(key_nodes.find_closest_node)
        agent = agent.replace(key_nodes = key_nodes.pos)
    else:
        key_nodes, find_key_node = None, None
        
    if FLAGS.sparse_data:
        dataset = d4rl_utils.sparse_data(dataset, sparse_data_index=sparse_data_index)

    pretrain_dataset = GCSDataset(dataset, find_key_node=find_key_node, **FLAGS.gcdataset.to_dict())

        
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
    elif 'Fetch' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    else:
        raise NotImplementedError

    base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
    observation = env.reset()
    
    if 'antmaze' in env_name:
        goal = env.wrapped_env.target_goal
        goal_info = base_observation.copy()
        goal_info[:2] = goal
    elif 'kitchen' in env_name:
        if 'visual' not in env_name:
            observation, goal_info = observation[:30].copy(), observation[30:].copy()
            goal_info[:9] = base_observation[:9]
    elif 'calvin' in env_name:
        observation = observation['ob']
        goal = np.array([0.25, 0.15, 0, 0.088, 1, 1])
        goal_info = base_observation.copy()
        goal_info[15:21] = goal
    elif 'Fetch' in env_name:
        state, _  = env.reset()
        base_observation, goal_info = state['observation'], state['desired_goal']
    else:
        raise NotImplementedError
    
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    debug = False
    if debug:
        model_path = '/home/spectrum/study/ASK_Baseline/ASK_Spherical_script/experiment_output/ant_ultra_diverse_hiql_06-25_09:55/antmaze-ultra-diverse-v0_sd000_06-25_09-55/params_200000.pkl'
        plot_value_map(agent, base_observation, goal_info, model_path)
        

    for i in tqdm.tqdm(range(1, total_steps + 1),
                   desc="main_train",
                   smoothing=0.1,
                   dynamic_ncols=True):
        pretrain_batch = pretrain_dataset.sample(FLAGS.batch_size)
        agent, update_info = supply_rng(agent.pretrain_update)(pretrain_batch)

        if i % FLAGS.log_interval == 0:
            debug_statistics = get_debug_statistics(agent, pretrain_batch)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)
                
        if FLAGS.use_rep=="vae_encoder" and FLAGS.config['build_keynode_time']=="during_training" and not(i % FLAGS.eval_interval == 0) :
            encoder_fn = jax.jit(jax.vmap(agent.get_vae_state_rep))
            rep_observations = d4rl_utils.get_rep_observation(encoder_fn, dataset, FLAGS)
            dataset = d4rl_utils.add_data(dataset, rep_observations)
            key_nodes.construct_nodes(rep_observations=rep_observations)                
            find_key_node = jax.jit(key_nodes.find_closest_node)
            pretrain_dataset = GCSDataset(dataset, find_key_node = find_key_node, encoder_fn=encoder_fn, **FLAGS.gcdataset.to_dict())

        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.use_rep in ["hilp_subgoal_encoder", "hilp_encoder"]:
                encoder_fn = jax.jit(jax.vmap(agent.get_hilp_phi))
                rep_observations = d4rl_utils.get_hilp_rep_observation(encoder_fn, dataset, FLAGS)
            elif FLAGS.use_rep == "hiql_goal_encoder":
                encoder_fn = jax.jit(jax.vmap(agent.get_value_goal))
                # 0610 승호수정 spherical
                if FLAGS.rep_type == 'concat':
                    rep_observations = d4rl_utils.get_rep_observation_spherical(encoder_fn, dataset, FLAGS)
                elif FLAGS.rep_type == 'state':
                    rep_observations = d4rl_utils.get_rep_observation_goal_only(encoder_fn, dataset, FLAGS)
            elif FLAGS.use_rep != '0':
                if FLAGS.kmean_weight_type == 'hilbert_td' :
                    dataset = d4rl_utils.hilp_add_data(dataset, rep_observations)
                elif FLAGS.kmean_weight_type in ['rtg_discount', 'rtg_uniform']:
                    dataset = d4rl_utils.add_data(dataset, rep_observations)
                    
            if FLAGS.use_rep in ["hiql_goal_encoder", "hilp_subgoal_encoder", "hilp_encoder"]:
                # key_nodes, sparse_data_index = keynode_utils.build_keynodes(dataset, flags=FLAGS, episode_index= episode_index)
                # 0610 승호수정 spherical
                key_nodes.construct_nodes(rep_observations=rep_observations, spherical_On=FLAGS.spherical_On)
                find_key_node = jax.jit(key_nodes.find_closest_node)
                agent = agent.replace(key_nodes = key_nodes.pos)
                pretrain_dataset = GCSDataset(dataset, find_key_node = find_key_node, **FLAGS.gcdataset.to_dict())
            
            eval_episodes = 1 if i == 1 else FLAGS.eval_episodes
            num_video_episodes = 0 if i == 1 else FLAGS.num_video_episodes
            
            policy_fn = partial(supply_rng(agent.sample_actions))
            high_policy_fn = partial(supply_rng(agent.sample_high_actions))
            base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
            if FLAGS.use_rep=="hiql_goal_encoder":
                value_goal_fn = jax.jit(agent.get_value_goal)
            elif FLAGS.use_rep=="vae_encoder":
                encoder_fn = jax.jit(agent.get_vae_state_rep)
                decoder_fn = jax.jit(agent.get_vae_rep_state)
                value_goal_fn = jax.jit(agent.get_value_goal)
            
            # policy_fn = jax.jit(policy_fn)
            # high_policy_fn = jax.jit(high_policy_fn)
            value_goal_fn = jax.jit(agent.get_value_goal)
            # encoder_fn = jax.jit(agent.get_vae_state_rep)

            eval_info, trajs, renders, rep_trajectories, cos_distances = evaluate_with_trajectories(
                    policy_fn=policy_fn, high_policy_fn=high_policy_fn, encoder_fn=encoder_fn, decoder_fn=decoder_fn, value_goal_fn=value_goal_fn, env=env,
                    env_name=FLAGS.env_name, num_episodes=eval_episodes,
                    base_observation=base_observation, goal_info=goal_info, num_video_episodes=num_video_episodes,
                    eval_temperature=0,
                    config=FLAGS.config,
                    find_key_node=find_key_node,
                    key_nodes=key_nodes,
                    FLAGS=FLAGS
                )
            value_map = plot_value_map(agent, base_observation, goal_info, i, g_start_time)
            
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            eval_metrics['value_map'] = wandb.Image(value_map)
            
            if FLAGS.num_video_episodes > 0 and len(renders):
                video = record_video('Video', i, renders=renders)
                eval_metrics['video'] = video
            # traj_metrics = get_traj_v(agent, example_trajectory)
            # value_viz = viz_utils.make_visual_no_image(
            #     traj_metrics,
            #     [partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
            # )
            # eval_metrics['value_traj_viz'] = wandb.Image(value_viz)
            if 'antmaze' in FLAGS.env_name and 'large' in FLAGS.env_name and FLAGS.env_name.startswith('antmaze'):
                traj_image = d4rl_ant.trajectory_image(viz_env, viz_dataset, trajs)
                eval_metrics['trajectories'] = wandb.Image(traj_image)
                new_metrics_dist = viz.get_distance_metrics(trajs)
                eval_metrics.update({
                    f'debugging/{k}': v for k, v in new_metrics_dist.items()})
                image_v = d4rl_ant.gcvalue_image(
                    viz_env,
                    viz_dataset,
                    partial(get_v, agent),
                )
                eval_metrics['v'] = wandb.Image(image_v)

            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
                config=FLAGS.config.to_dict()
            )
            if i == 1 or i % FLAGS.save_interval == 0:
                if FLAGS.use_rep !='0':
                    rep_trajectory_fname = os.path.join(FLAGS.save_dir, f'rep_trajectories_{i}.pkl')
                    with open(rep_trajectory_fname, 'wb') as f:
                        pickle.dump(np.array(rep_trajectories), f)
                    all_state_fname = os.path.join(FLAGS.save_dir, f'all_state_{i}.pkl')
                    with open(all_state_fname, 'wb') as f:
                        pickle.dump(np.array(rep_observations), f)

                if FLAGS.spherical_On:
                    cos_distances_path = os.path.join(FLAGS.save_dir, f'cos_distances_{i}.pkl')
                    with open(cos_distances_path, 'wb') as f:
                        pickle.dump(np.array(cos_distances), f)
                
                key_node_fname = os.path.join(FLAGS.save_dir, f'key_node_{i}.pkl')
                with open(key_node_fname, 'wb') as f:
                    pickle.dump(np.array(key_nodes.pos), f)
                
                fname = os.path.join(FLAGS.save_dir, f'params_{i}.pkl')
                print(f'Saving to {fname}')
                with open(fname, "wb") as f:
                    pickle.dump(save_dict, f)         
            if 'calvin' in FLAGS.env_name:
                score = eval_metrics['evaluation/final.return']
            else:
                score = eval_metrics['evaluation/episode.return']
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
            
    train_logger.close()
    eval_logger.close()

if __name__ == '__main__':
    import random
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    
    app.run(main)
