import gym
import d4rl
import numpy as np

from jaxrl_m.dataset import Dataset
from jaxrl_m.evaluation import EpisodeMonitor

import jax.numpy as jnp


def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env

def get_dataset(env: gym.Env,
                env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                dataset=None,
                filter_terminals=False,
                obs_dtype=np.float32,
                flag=None
                ):
        # goal_info = None
        if dataset is None:
            dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dataset['terminals'][-1] = 1
        # kitchen 환경에서만 filter_terminals=True
        if filter_terminals: 
            # drop terminal transitions
            non_last_idx = np.nonzero(~dataset['terminals'])[0]
            last_idx = np.nonzero(dataset['terminals'])[0]
            penult_idx = last_idx - 1
            new_dataset = dict()
            for k, v in dataset.items():
                if k == 'terminals':
                    v[penult_idx] = 1
                new_dataset[k] = v[non_last_idx]
            dataset = new_dataset

        if 'antmaze' in env_name:
            # antmaze: terminals are incorrect for GCRL
            dones_float = np.zeros_like(dataset['rewards'])
            dataset['terminals'][:] = 0.
            for i in range(len(dones_float) - 1):
                if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
            dones_float[-1] = 1
            dataset['rewards'], goal_info = relabel_ant(env, env_name, dataset, flag) # flags.use_goal_info_On 일때만, goal_info 반환하고 나머지는 None
        elif 'calvin' in env_name:
            goal_info, episode_index = relabel_calvin(env, env_name, dataset, flag)
            dones_float = dataset['terminals'].copy()
        elif 'kitchen' in env_name:
            dones_float = dataset['terminals'].copy()
            goal_info, episode_index = relabel_kitchen(env, env_name, dataset, flag) 

        else:
            NotImplementedError
            
        observations = dataset['observations'].astype(obs_dtype)
        next_observations = dataset['next_observations'].astype(obs_dtype)

        # if 'ant' in env_name:
        #     if flag.kmean_weight_type == 'rtg_discount':
        #         returns, episode_index = calc_return_to_go_ant(dataset)
        #     elif flag.kmean_weight_type == 'rtg_uniform':
        #         returns, episode_index = calc_return_to_go_ant_trajectory(dataset)
        #     elif flag.kmean_weight_type == 'hilbert_td':      
        #         returns, episode_index = None, None
            
        # elif 'kitchen' in env_name:
        #     returns, episode_index = calc_return_to_go_kitchen(dataset, flag)
        # elif 'calvin' in env_name:
        #     returns, episode_index = calc_return_to_go_calvin(dataset, flag)
        return Dataset.create(
            observations=observations,
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=1.0 - dones_float.astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=next_observations,
            # returns = returns,
            goal_info = goal_info,
            )
        # episode_index
            
def relabel_ant(env, env_name, dataset, flags):
    observation_pos = dataset['observations'].reshape(999,1000,-1)[:,:,:2]  
    new_rewards = np.zeros((999,1000))  
    # goal_info = None
    # if flags.use_goal_info_On:
    #     d4rl_dataset = env.get_dataset()
    #     goal_pos = d4rl_dataset['infos/goal']  
    #     goal_pos = goal_pos[:999*1001].reshape(999,1001,2)[:,:1000,:]
    #     new_rewards = np.where(np.linalg.norm(observation_pos.reshape(999,1000,2) - goal_pos, axis=-1)<1, 1, 0)
    #     new_rewards = new_rewards.reshape(-1).astype(np.float32)
    #     goal_info = goal_pos.reshape(999*1000, 2)
    if 'ultra' in env_name:
        # unique_goal_pos = np.array([[12,0], [40,0], [52,0], [0,16], [0,28], [32,36], [52,0], [52,16], [52,36]])
        last_obs = observation_pos[:,-1,:2]
        for idx, obs_pos in enumerate(observation_pos.transpose(1,0,2)):
            distance = np.linalg.norm(last_obs - obs_pos, axis=1)
            index = np.where(distance <= 2.5, True, False)
            new_rewards[index, idx] = 1.0
        reshaped_obs = dataset['observations'].reshape(999,1000,-1)
        goal_info = np.repeat(reshaped_obs[:,-1], repeats=1000, axis=0)
        
    # elif 'large' in env_name:
    #     unique_goal_pos = np.array([[0,0],[0,12], [0,24], [12,22], [12,8], [20,16], [36,0], [32.75,24.75], [32,16]])
    #     for idx, obs_pos in enumerate(observation_pos):
    #         distance = np.linalg.norm(unique_goal_pos - obs_pos, axis=1)
    #         index = np.where(distance <= 1, 1, 0)
    #         if any(index):
    #             new_rewards[idx] = 1.0
    # elif 'medium' in env_name:
    #     unique_goal_pos = np.array([[20.5, 20.5]])
    #     for idx, obs_pos in enumerate(observation_pos):
    #         distance = np.linalg.norm(unique_goal_pos - obs_pos, axis=1)
    #         index = np.where(distance <= 1, 1, 0)
    #         if any(index):
    #             new_rewards[idx] = 1.0 
    return new_rewards.reshape(-1), goal_info

# 질문: 승호 나중에 코드 체크
def relabel_calvin(env, env_name, dataset, flags):
    t = -1
    for i, d in enumerate(dataset['observations'][:,15:21]):
        if dataset['episodes'][i] != t:
            t = dataset['episodes'][i] 
            start_door, start_drawer, _, _, start_lightbulb, start_led = d
        door, drawer, _, _, lightbulb, led = d
        if abs(start_drawer - drawer) > 0.12:
            dataset['rewards'][i] +=1
        if start_lightbulb != lightbulb:
            dataset['rewards'][i] +=1
        if abs(start_door - door) > 0.15:
            dataset['rewards'][i] +=1
        if start_led != led:
            dataset['rewards'][i] +=1
    reshape_obs = dataset['observations'].reshape(1204,499,-1)[:,-1,:]
    goal_info = np.repeat(reshape_obs[:,np.newaxis,:], 499, axis=1).reshape(-1, 21)
    return goal_info, dataset['episodes']

def relabel_kitchen(env, env_name, dataset, flags):
    episode_index = []
    tmp_episode = []
    episodes = []
    tmp_episode_cnt = 0

    for i, t in enumerate(dataset['terminals']):
        tmp_episode_cnt+=1
        tmp_episode.append(i)            
        if t:
            episodes.extend(np.tile(dataset['observations'][i][:30], (tmp_episode_cnt,1)))
            episode_index.append(tmp_episode)
            tmp_episode = []   
            tmp_episode_cnt = 0          
            
    goal_info = np.stack(episodes)

    return goal_info, episode_index


def calc_return_to_go_ant_trajectory(dataset):
    rewards = dataset['rewards'].reshape(999, -1)   
    return_to_go = np.zeros(rewards.shape, dtype=np.float32)
    # goal에 최초로 도착한 step 찾기
    # episode_index : reward 발생한 episode index
    # state_index : reward를 얻은 episode 내 step index
    episode_index, state_index = np.where(rewards)
    # episode 내에서 reward를 얻은 state가 여러개일경우, reward를 얻은 episode index의 첫번째 index를 사용해서, 1개의 episode내에서 최초로 reward를 얻은 state의 index를 찾음
    first_episode_index, first_state_index = np.unique(episode_index, return_index=True)
    obs = dataset['observations'].reshape(999,1000,-1)
    distance = np.linalg.norm(obs[first_episode_index, 0, :2] - obs[first_episode_index, state_index[first_state_index],:2], ord=1, axis=1)
    distance = distance /  (distance.max() - distance.min())
    normalized_weight = 1000/ state_index[first_state_index] * distance
    return_to_go += normalized_weight.min()
    return_to_go[first_episode_index] += normalized_weight.reshape(-1,1)    
    return return_to_go.reshape(-1).astype(np.float32), np.arange(len(dataset['observations'])).reshape(999,-1)

def calc_return_to_go_ant(dataset):
    gamma=0.998
    rewards = dataset['rewards'].reshape(999, -1) # ultra:(999,1000)
    terminals = dataset['rewards'].reshape(999, -1) 
    return_to_go = np.zeros(rewards.shape, dtype=np.float32)
    prev_return = 0    
    for i in range(return_to_go.shape[1]):
        return_to_go[:,-i-1] = rewards[:,-i-1] + gamma * prev_return * (1 - terminals[:,-i-1])
        prev_return = return_to_go[:,-i-1]
    return return_to_go.reshape(-1), np.arange(len(dataset['observations'])).reshape(999,-1)

def calc_return_to_go_kitchen(dataset, flags):
    def find_episode_index():
        episode_index = []
        expert_episode_index = []
        tmp_episode = []
        threshold = 3 if flags.expert_data_On else 1
        for i, r in enumerate(dataset['rewards']):
            tmp_episode.append(i)
            if dataset['terminals'][i]:
                if dataset['rewards'][i] >= threshold:
                    expert_episode_index.append(np.array(tmp_episode))
                episode_index.append(np.array(tmp_episode))
                tmp_episode = []
        return expert_episode_index if flags.expert_data_On else episode_index
    
    gamma=0.9
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    return_to_go = np.zeros(rewards.shape, dtype=np.float32)
    episode_index = find_episode_index() 
    prev_return = 0
    for e in episode_index:
        for i in e[::-1]:
            return_to_go[i] = rewards[i] + gamma * prev_return * (1 - terminals[i])
            prev_return = return_to_go[i] 
    return return_to_go, episode_index

# 질문: 승호 나중에 코드 체크
def calc_return_to_go_calvin(dataset, flags):
    gamma=0.9
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    return_to_go = np.zeros(rewards.shape, dtype=np.float32)
    def find_episode_index():
        episode_index = []
        expert_episode_index = []
        tmp_episode = []
        t = -1
        for i, r in enumerate(dataset['rewards']):
            if t != dataset['episodes'][i]:
                if t != -1 and dataset['rewards'][i-1] >=4:
                    expert_episode_index.append(np.array(tmp_episode))
                episode_index.append(np.array(tmp_episode))
                tmp_episode = []
                t = dataset['episodes'][i]
            tmp_episode.append(i)
        
        if flags.expert_data_On or dataset['rewards'][-1] >= 4:
            expert_episode_index.append(np.array(tmp_episode))
        episode_index.append(np.array(tmp_episode))

        if episode_index and len(episode_index[0]) == 0:
            episode_index.pop(0)
            
        return episode_index, expert_episode_index
    def change_reward_sparse():
        sparse_reward = [0]
        for i, r in enumerate(dataset['rewards'][1:]):
            if dataset['rewards'][i]==dataset['rewards'][i-1]:
                sparse_reward.append(0)
            else:
                sparse_reward.append(1)
        return sparse_reward
    # sparse reward 적용시 reward -> sparse_reward로 변경
    # reward = change_reward_sparse() # reward를 1번씩만 받도록 변경
    episode_index, expert_episode_index = find_episode_index() # reward가 3이상인 episode의 index를 찾음
    prev_return = 0
    for e in expert_episode_index:
        for i in e[::-1]:
            return_to_go[i] = rewards[i] + gamma * prev_return * (1 - terminals[i])
            prev_return = return_to_go[i]
    return return_to_go.astype(np.float32), episode_index

def sparse_data(dataset, sparse_data_index=None):
    if 'rep_observations' in dataset and 'rep_next_observations' in dataset:
        return Dataset.create(
            observations=dataset['observations'][sparse_data_index],
            actions=dataset['actions'][sparse_data_index].astype(np.float32),
            rewards=dataset['rewards'][sparse_data_index].astype(np.float32),
            masks=1.0 - dataset['dones_float'][sparse_data_index].astype(np.float32),
            dones_float=dataset['dones_float'][sparse_data_index].astype(np.float32),
            next_observations=dataset['next_observations'][sparse_data_index],
            returns = dataset['returns'][sparse_data_index],
            goal_info = dataset['goal_info'],
            rep_observations = dataset['rep_observations'][sparse_data_index],
            rep_next_observations = dataset['rep_next_observations'][sparse_data_index]
        )
    else:
        return Dataset.create(
            observations=dataset['observations'][sparse_data_index],
            actions=dataset['actions'][sparse_data_index].astype(np.float32),
            rewards=dataset['rewards'][sparse_data_index].astype(np.float32),
            masks=1.0 - dataset['dones_float'][sparse_data_index].astype(np.float32),
            dones_float=dataset['dones_float'][sparse_data_index].astype(np.float32),
            next_observations=dataset['next_observations'][sparse_data_index],
            returns = dataset['returns'][sparse_data_index],
            goal_info = dataset['goal_info']
        )
        
def add_data(dataset, rep_observations=None, rep_next_observations=None, key_node=None):
    return Dataset.create(
            observations=dataset['observations'],
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=1.0 - dataset['dones_float'].astype(np.float32),
            dones_float=dataset['dones_float'].astype(np.float32),
            next_observations=dataset['next_observations'],
            # returns = dataset['returns'],
            goal_info = dataset['goal_info'],
            rep_observations = rep_observations,
            rep_next_observations = rep_next_observations,
            key_node=key_node
        )
    
def get_rep_observation(encoder_fn, dataset, FLAGS, goal=None):
    mini_batch = 50000
    size = len(dataset['observations']) // mini_batch
    rep_observations = np.zeros((len(dataset['observations']), FLAGS.rep_dim), dtype=np.float32)
    if 'ant' in FLAGS.env_name:
        goal = dataset['observations'].copy()
        goal[:,:2] = dataset['goal_info'] # 현재 flags.use_goal_info_On 일때만, goal_info 반환하고 나머지는 None임으로 주의
    # 질문: 대충 goal 하나 설정 (kitchen 수정 필요)
    elif 'kitchen' in FLAGS.env_name:
        goal = dataset['observations'][0]
        goal = np.repeat(goal[np.newaxis,:], len(dataset['observations']), axis=0)
    for i in range(size+1):
        if goal is not None:
            rep_observations[mini_batch*i:mini_batch*(i+1)] = encoder_fn(bases=dataset['observations'][mini_batch*i:mini_batch*(i+1)], targets=goal[mini_batch*i:mini_batch*(i+1)])
        else:
            rep_observations[mini_batch*i:mini_batch*(i+1)] = encoder_fn(targets=dataset['observations'][mini_batch*i:mini_batch*(i+1)])
    return rep_observations 

# 0610 승호수정 goal only
def get_rep_observation_goal_only(encoder_fn, dataset, FLAGS):
    mini_batch = 50000
    size = len(dataset['observations']) // mini_batch
    rep_observations = np.zeros((len(dataset['observations']), FLAGS.rep_dim), dtype=np.float32)
    for i in range(size+1):
        rep_observations[mini_batch*i:mini_batch*(i+1)] = encoder_fn(bases=dataset['observations'][mini_batch*i:mini_batch*(i+1)], targets=dataset['observations'][mini_batch*i:mini_batch*(i+1)])
    return rep_observations 

# 0610 승호수정 spherical
def get_rep_observation_spherical(encoder_fn, dataset, FLAGS):
    observations = dataset['observations'].reshape(-1, 1000, dataset['observations'].shape[-1])  # antmaze 기준 (999, 1000, 29)
    rep_observations = np.zeros((observations.shape[0] * observations.shape[1], FLAGS.rep_dim), dtype=np.float32)

    bases_indx = np.arange(1000)  # 0부터 999까지의 인덱스
    
    # 각 트레젝토리에 대해 연산 수행 (배치 처리)
    for traj_index in range(len(observations)):
        traj = observations[traj_index]
        bases=traj[bases_indx]   
        targets = np.array([traj[min(i + FLAGS.way_steps, 999)] for i in bases_indx])  # targets 배열 생성
        
        rep_observations_batch = encoder_fn(bases=bases, targets=targets)
        rep_observations[traj_index*1000 : traj_index*1000+len(bases_indx)] = rep_observations_batch
    
    return rep_observations

def get_hilp_rep_observation(encoder_fn, dataset, FLAGS, goal=None):
    mini_batch = 50000
    size = len(dataset['observations']) // mini_batch
    rep_observations = np.zeros((len(dataset['observations']), FLAGS.rep_dim), dtype=np.float32)
    for i in range(size+1):
        rep_observations[mini_batch*i:mini_batch*(i+1)] = encoder_fn(observations=dataset['observations'][mini_batch*i:mini_batch*(i+1)])
    return rep_observations 

def hilp_add_data(dataset, rep_observations):
    reshape_obs = rep_observations.reshape(999, 1000, rep_observations.shape[-1])

    diff = reshape_obs[:, 1:, :] - reshape_obs[:, :-1, :]
    squared_dist = jnp.sum(diff ** 2, axis=-1)
    squared_dist = jnp.pad(squared_dist, ((0, 0), (1, 0)), constant_values=0)
    td_value = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
    td_value = np.asarray(td_value.reshape(-1).astype(jnp.float32))
    
    return Dataset.create(
            observations=dataset['observations'],
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=1.0 - dataset['dones_float'].astype(np.float32),
            dones_float=dataset['dones_float'].astype(np.float32),
            next_observations=dataset['next_observations'],
            returns = td_value,
            goal_info = dataset['goal_info'],
        )
    
def get_transition_index(agent, dataset, flags):
    if 'ant' in flags.env_name:
        return get_transition_index_ant(agent, dataset, flags)
    elif 'kitchen' in flags.env_name:
        return get_transition_index_kitchen(agent, dataset, flags)
    else:
        NotImplementedError

def get_hilp_obs(agent, observations, flags):
    import jax
    rep_obs = np.zeros((len(observations), flags.hilp_skill_dim), dtype=np.float32)
    # encoder_fn = jax.jit(agent.get_hilp_phi)
    encoder_fn = agent.get_hilp_phi
    # filtered_observations = observations[transition_index]
    mini_batch = 50000
    size = len(observations) // mini_batch
    for i in range(size+1):
        rep_obs[mini_batch*i:mini_batch*(i+1)] = encoder_fn(observations=observations[mini_batch*i:mini_batch*(i+1)])
    
    return rep_obs

def get_latent_key_nodes(find_key_node_in_dataset, observations, flags):
    # key_node, letent_key_node = find_key_node_in_dataset(observations)
    key_node = np.zeros((len(observations),flags.config['obs_dim']), dtype=np.float32)
    letent_key_node = np.zeros((len(observations), flags.hilp_skill_dim), dtype=np.float32)
    
    mini_batch = 50000
    size = len(observations) // mini_batch
    for i in range(size+1):
        key_node[mini_batch*i:mini_batch*(i+1)], letent_key_node[mini_batch*i:mini_batch*(i+1)] = find_key_node_in_dataset(observations[mini_batch*i:mini_batch*(i+1)])
    
    return key_node, letent_key_node


def get_transition_index_ant(agent, dataset, flags):
    import jax
    mini_batch = 50000
    size = len(dataset['observations']) // mini_batch
    values = np.zeros(len(dataset['observations']), dtype=np.float32)
    value_fn = jax.jit(agent.get_hilp_value)
    for i in range(size+1):
        values[mini_batch*i:mini_batch*(i+1)] = value_fn(observations=dataset['observations'][mini_batch*i:mini_batch*(i+1)], goals=dataset['goal_info'][mini_batch*i:mini_batch*(i+1)])[0]
    
            
    # rewards = dataset['rewards'].reshape(999,1000)
    # dones_index = np.argmax(rewards, axis=1)
    # dones_index[dones_index==0] = 999
    
    reshape_values = values.reshape(999, 1000)
    tdd_ = reshape_values[:, 1:] - reshape_values[:, :-1]
    tdd = jnp.pad(tdd_, ((0, 0), (1, 0)), constant_values=tdd_[tdd_>0].mean())
    # dones_indexes = np.ones_like(reshape_values, dtype=bool)
 
    # for i, t in enumerate(tdd):
    #     dones_indexes[i,dones_index[i]:] = False
    td_flatten = np.array(tdd, dtype=np.float32).reshape(-1)
    # tdd = tdd.at[dones_indexes==False].set(0)
    # hlip_filtered_index = np.where(tdd > tdd[tdd>0].mean(), True, False)
    
    index = np.zeros_like(reshape_values, dtype=bool)
    max_values = reshape_values[:,0]
    for i, t in enumerate(reshape_values[:,1:]):
        better_idx = np.where(max_values < reshape_values[:,i+1], True, False)
        max_values[better_idx] = reshape_values[better_idx, i+1]
        
        # positive_idx = np.where(reshape_values[:,i+1]-reshape_values[:,i] >0, True, False)
        # idx = better_idx * positive_idx
        index[better_idx,i+1] = True
    
    hlip_filtered_index = index.reshape(-1)
    
    # tdd = tdd[index]
    # filtered_transition_index = hlip_filtered_index * dones_indexes
    # hlip_filtered_index = hlip_filtered_index * dones_indexes
    
    
    # filtered_transition_index = hlip_filtered_index * dones_indexes * index
    # tdd = tdd[hlip_filtered_index]
    # print(f'before filtered tdds {tdd[0]=}, {tdd.min()=}, {tdd.max()=}, {hlip_filtered_index.mean()=}, {td_flatten[hlip_filtered_index].mean()=}')
    # print(f'after filtered tdds {td_flatten[0,0]=}, {td_flatten.min()=}, {td_flatten.max()=}, {hlip_filtered_index.mean()*100=:.2f} % used dataset, {td_flatten[hlip_filtered_index].mean()=}')
    print(f'after filtered tdds {td_flatten[0]=}, {td_flatten.min()=}, {td_flatten.max()=}, {hlip_filtered_index.mean()*100=:.2f} % used dataset, {td_flatten[hlip_filtered_index].mean()=}')
    print(f'better filtered tdds {td_flatten[0]=}, {td_flatten[hlip_filtered_index].min()=}, {td_flatten[hlip_filtered_index].max()=}, {hlip_filtered_index.mean()*100=:.2f} % used dataset, {td_flatten[hlip_filtered_index].mean()=}')
    
    obs_index = np.random.choice(len(dataset['observations']), size=hlip_filtered_index.sum())
    plot_obs(dataset['observations'][obs_index], path='all')
    plot_obs(dataset['observations'][hlip_filtered_index.reshape(-1).astype(bool)], path='filtered')
    
    
    # return filtered_transition_index.reshape(-1).astype(bool), hlip_filtered_index.reshape(-1).astype(bool), dones_indexes.reshape(-1)
    return hlip_filtered_index.reshape(-1).astype(bool), hlip_filtered_index.reshape(-1).astype(bool), None

def get_transition_index_kitchen(agent, dataset, flags):
    import jax
    mini_batch = 50000
    size = len(dataset['observations']) // mini_batch
    values = np.zeros(len(dataset['observations']), dtype=np.float32)
    value_fn = jax.jit(agent.get_hilp_value)
    
    for i in range(size+1):
        values[mini_batch*i:mini_batch*(i+1)] = value_fn(observations=dataset['observations'][mini_batch*i:mini_batch*(i+1)], goals=dataset['goal_info'][mini_batch*i:mini_batch*(i+1)])[0]
    #####################################################################################
    # tdd_ = values[1:] - values[:-1]
    # first_tdd = jnp.array([tdd_[tdd_>0].mean()])
    # tdd = jnp.concatenate([first_tdd, tdd_],axis=0)
    # hlip_filtered_index = np.where(tdd > tdd[tdd>0].mean(), True, False)
    # dones_indexes = np.where(dataset['dones_float']==1, True, False)
    
    # hlip_filtered_index = hlip_filtered_index.reshape(-1).astype(bool)
    
    # print(f'{tdd[0]=}, {tdd.min()=}, {tdd.max()=}, {hlip_filtered_index.mean()=},, {tdd[hlip_filtered_index].mean()=}')

    # return hlip_filtered_index, hlip_filtered_index, dones_indexes
    
    #####################################################################################
    max_values = values[0]
    hlip_filtered_index = np.zeros_like(values, dtype=bool)
    for i, v in enumerate(values[:-1]):
        if v > max_values:
            max_values = v
            hlip_filtered_index[i] = True
        if dataset['dones_float'][i]:
            max_values = values[i+1]
    


    
    tdd_ = values[1:] - values[:-1]
    first_tdd = jnp.array([tdd_[tdd_>0].mean()])
    td_flatten = jnp.concatenate([first_tdd, tdd_],axis=0)
    # hlip_filtered_index = np.where(td_flatten > td_flatten[td_flatten>0].mean(), True, False)
 
    # index = np.zeros_like(values, dtype=bool)
    # max_values = values[:,0]
    # for i, t in enumerate(values[:,1:]):
    #     better_idx = np.where(max_values < values[:,i+1], True, False)
    #     max_values[better_idx] = values[better_idx, i+1]
    #     index[better_idx,i+1] = True
    
    # hlip_filtered_index = index.reshape(-1)

    print(f'after filtered tdds {td_flatten[0]=}, {td_flatten.min()=}, {td_flatten.max()=}, {hlip_filtered_index.mean()*100=:.2f} % used dataset, {td_flatten[hlip_filtered_index].mean()=}')
    print(f'better filtered tdds {td_flatten[0]=}, {td_flatten[hlip_filtered_index].min()=}, {td_flatten[hlip_filtered_index].max()=}, {hlip_filtered_index.mean()*100=:.2f} % used dataset, {td_flatten[hlip_filtered_index].mean()=}')
    
    return hlip_filtered_index, hlip_filtered_index, None
    
    

def plot_obs(observations, s=1, xlim=55, ylim=40, path=None ,c=None):
    if path is None:
        path = 0
        
    x, y = observations[:, 0], observations[:, 1]
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(10,8))
    print(f'{x.shape=}, {y.shape}')
    if c is not None:
        cmap = plt.cm.bwr
        plt.scatter(x,y, s=s, cmap=cmap, c=c)
        plt.colorbar()
    else:
        plt.scatter(x,y, s=s)
    plt.xlim([-2,xlim])
    plt.ylim([-2,ylim])
    import time
    g_start_time = time.strftime('%m-%d_%H-%M')
    import os
    dir_pth = os.path.dirname(os.path.dirname(__file__))
    os.makedirs(os.path.join(dir_pth , f'plot_ant/{g_start_time}'), exist_ok=True)
    plt.savefig(os.path.join(dir_pth , f'plot_ant/{g_start_time}/{path}_obs.png'), format="PNG", dpi=300)
    plt.close()
    
    