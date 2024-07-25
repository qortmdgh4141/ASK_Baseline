from typing import Dict
import jax
import gym
import numpy as np
from collections import defaultdict
import time
import tqdm
import jax.numpy as jnp

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped

def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img

def evaluate_with_trajectories(
        policy_fn, high_policy_fn, env: gym.Env, env_name, num_episodes: int, base_observation=None, num_video_episodes=0,
        eval_temperature=0, epsilon=0, 
        config=None, find_key_node = None, hilp_fn=None, FLAGS = None, distance_fn=None, nodes=None
) -> Dict[str, float]:
    trajectories = []
    stats = defaultdict(list)

    renders = []
    rep_trajectories = []
    for i in tqdm.tqdm(range(num_episodes + num_video_episodes), desc="evaluate_with_trajectories"):
        trajectory = defaultdict(list)
        observation, done = env.reset(), False
        # Set goal
        if 'antmaze' in env_name:
            goal = env.wrapped_env.target_goal
            obs_goal = base_observation.copy()
            obs_goal[:2] = goal
            node_dim = np.arange(2)
            interval, min_dist = 10, 4
        elif 'kitchen' in env_name:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
            node_dim = np.arange(9)
            interval, min_dist = 6, 3.0
        elif 'calvin' in env_name:
            observation = observation['ob']
            goal = np.array([0.25, 0.15, 0, 0.088, 1, 1])
            obs_goal = base_observation.copy()
            obs_goal[15:21] = goal
            node_dim = np.arange(15)
            interval, min_dist = 3, 2.0 
        else:
            raise NotImplementedError
        
        render = []
        rep_trajectory = []
        h_steps = []
        dists = []
        diff_sub_goal_nodes = []
        step = 0
        h_step = 1
        interval = interval if FLAGS.relative_dist_in_eval_On else 1
        dist = 0
        init_dist = 1e5 if FLAGS.relative_dist_in_eval_On else 0
        cos_distances = []
        
            
        while not done:
               
            # if h_step == interval or dist < init_dist * 0.5:
            if h_step == interval or step==0:
                cur_obs_subgoal = high_policy_fn(observations=observation, goals=obs_goal, temperature=0)
                if FLAGS.high_action_in_hilp or 'cql' in FLAGS.algo_name:
                    cur_obs_goal = plot_subgoal = cur_obs_subgoal
                else:
                    cur_obs_goal = plot_subgoal = observation + cur_obs_subgoal
                if config['use_keynode_in_eval_On']:
                    # cur_obs_subgoal = high_policy_fn(observations=observation, goals=obs_goal, temperature=0, num_samples=1)
                    if FLAGS.high_action_in_hilp:
                        distance, cur_obs_latent_key_node, hilp_subgoal, cur_obs_key_node  = find_key_node(jnp.concatenate([hilp_fn(observations=observation), cur_obs_subgoal], axis=0))
                        cur_obs_goal = cur_obs_latent_key_node
                    else:
                        distance, _, hilp_subgoal, cur_obs_key_node  = find_key_node(hilp_fn(observations=jnp.vstack([observation, cur_obs_subgoal])).reshape(-1))
                        if cur_obs_key_node is None:
                            cur_obs_key_node = cur_obs_subgoal
                        cur_obs_goal = cur_obs_key_node
                        
                # if FLAGS.relative_dist_in_eval_On:
                #     init_dist = np.linalg.norm(cur_obs_subgoal - observation)
                

                # if config['use_keynode_in_eval_On']:
                    # distance, nodes, centroids, cur_obs_key_node = find_key_node(cur_obs_sub_goal)
                    # cur_obs_key_node = nodes.find_centroid(cur_obs_sub_goal.reshape(1,-1))
                    # cos_distance, _, _, cur_obs_key_node = nodes.find_centroid(cur_obs_sub_goal.reshape(1,-1))
                    # if cos_distance >= FLAGS.mapping_threshold:
                    #     diff_sub_goal_node = np.linalg.norm(cur_obs_key_node - cur_obs_sub_goal, axis=-1, keepdims=True)
                    #     diff_sub_goal_nodes.append(diff_sub_goal_node)
                    # cos_distances.append(cos_distance)

                dist = np.linalg.norm(cur_obs_goal[node_dim] - observation[node_dim])
                h_step = 0
                                       
            h_step +=1
            h_steps.append(h_step)
            dists.append(dist)
        
    
            if FLAGS.final_goal:
                action = policy_fn(observations=observation, goals=obs_goal, subgoals=cur_obs_goal, low_dim_goals=True, temperature=eval_temperature)
            else:
                action = policy_fn(observations=observation, goals=cur_obs_goal, low_dim_goals=True, temperature=eval_temperature)
            
            if 'antmaze' in env_name:
                next_observation, r, done, info = env.step(action)
            elif 'kitchen' in env_name:
                next_observation, r, done, info = env.step(action)
                next_observation = next_observation[:30]
            elif 'calvin' in env_name:
                next_observation, r, done, info = env.step({'ac': np.array(action)})
                next_observation = next_observation['ob']
                del info['robot_info']
                del info['scene_info']

            step += 1

            info['dists_mean'] = np.mean(dists)
            info['h_step_mean'] = np.mean(h_step)
 
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
                cur_obs_subgoal=cur_obs_goal
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
            
            # Render
            if i >= num_episodes and step % 3 == 0:

                if 'antmaze' in env_name:
                    size = 240
                    box_size = 0.015
                    cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
                    if ('large' in env_name or 'ultra' in env_name) and not FLAGS.high_action_in_hilp:
                        def xy_to_pixxy(x, y):
                            if 'large' in env_name:
                                pixx = (x / 36) * (0.93 - 0.07) + 0.07
                                pixy = (y / 24) * (0.21 - 0.79) + 0.79
                            elif 'ultra' in env_name:
                                pixx = (x / 52) * (0.955 - 0.05) + 0.05
                                pixy = (y / 36) * (0.19 - 0.81) + 0.81
                            return pixx, pixy

                        # subgoal : gray
                        x_sub_goal, y_sub_goal = plot_subgoal[:2]
                        sub_pixx, sub_pixy = xy_to_pixxy(x_sub_goal, y_sub_goal)
                        cur_frame[:3, int((sub_pixy - box_size) * size):int((sub_pixy + box_size) * size), int((sub_pixx - box_size) * size):int((sub_pixx + box_size) * size)] = 160
                        if config['use_keynode_in_eval_On']:
                            # key node : pink
                            x_node, y_node = cur_obs_key_node[:2]
                            node_pixx, node_pixy = xy_to_pixxy(x_node, y_node)
                            cur_frame[0, int((node_pixy - box_size) * size):int((node_pixy + box_size) * size), int((node_pixx - box_size) * size):int((node_pixx + box_size) * size)] = 255
                            cur_frame[2, int((node_pixy - box_size) * size):int((node_pixy + box_size) * size), int((node_pixx - box_size) * size):int((node_pixx + box_size) * size)] = 127
                    render.append(cur_frame)
                    
                elif 'kitchen' in env_name:
                    render.append(kitchen_render(env, wh=200).transpose(2, 0, 1))
                elif 'calvin' in env_name:
                    cur_frame = env.render(mode='rgb_array').transpose(2, 0, 1)
                    render.append(cur_frame)     
        if 'calvin' in env_name:
            info['return'] = sum(trajectory['reward'])
        add_to(stats, flatten(info, parent_key="final"))
        # trajectories.append(trajectory)
        if i >= num_episodes:
            renders.append(np.array(render))


    for k, v in stats.items():
        stats[k] = np.mean(v)
    
    return stats, trajectories, renders, rep_trajectories, cos_distances

class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )
                
        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self.total_timesteps = 0
        self._reset_stats()
        return self.env.reset()
