# from gymnasium_robotics.core import Wrapper
from jaxrl_m.dataset import Dataset
import numpy as np 

import gym
from gymnasium_robotics.envs.fetch import push
from gymnasium_robotics.envs.fetch import reach

def fetch_load(env_name, data):

    # observations = data['o'][:,:-1]
    # next_observations = data['o'][:,1:]
    # dones_float = np.zeros((data['o'].shape[0],data['o'].shape[1]-1))
    # dones_float[:,-1] = 1
    # dones_float = dones_float
    observations = data['o'][:,:-1].reshape(-1, data['o'].shape[-1])
    next_observations = data['o'][:,1:].reshape(-1, data['o'].shape[-1])
    dones_float = np.zeros((data['o'].shape[0],data['o'].shape[1]-1))
    dones_float[:,-1] = 1
    dones_float = dones_float.reshape(-1)
    config = {'observation_min':data['o'].min(),
            'observation_max':data['o'].max(),
            'observation_dim':data['o'].shape[-1],
            'action_min':data['u'].min(),
            'action_max':data['u'].max(),
            'action_dim':data['u'].shape[-1],
            }
    return Dataset.create(
            observations=observations.astype(np.float32),
            actions=data['u'].reshape(-1, data['u'].shape[-1]).astype(np.float32),
            # actions=data['u'].astype(np.float32),
            rewards=None,
            masks=None,
            dones_float=dones_float,
            next_observations=next_observations.astype(np.float32),
            returns = np.ones_like(dones_float).astype(np.float32),
            # goal_info = data['g'],
            goal_info = data['g'].reshape(-1, data['g'].shape[-1]),
            ), None, config

def reach_goal(env_name, state, desired_goal):
    epsilon = 0.05
    if 'Reach' in env_name:
        achieved_goal = state[:3]
    else:
        achieved_goal = state[3:6]
        

    return np.linalg.norm((achieved_goal, desired_goal)) < epsilon

# class FetchGoalWrapper(Wrapper):
#     def __init__(self, env, env_name):
#         Wrapper.__init__(self, env)
#         self.env = env
#         self.env_name = env_name
#         self.action_space = env.action_space
#         self.observation_space = env.observation_space
    
#     def reset(self):
#         return self.env.reset()
    
#     def compute_rewards(self, achieved_goal, desired_goal, info=None):
#         return self.env.compute_reward(self.achieved_goal, self.desired_goal, self.info)

#     def step(self, action):
#         state, reward, done, truncated, self.info = self.env.step(action)
#         # Fetch default spase reward : success = 0, fail = -1 
#         # reward + 1 --> success = 1, fail = 0
#         reward +=1 
#         self.observation, self.achieved_goal, self.desired_goal = state['observation'], state['achieved_goal'], state['desired_goal']
        
#         if reach_goal(self.env_name, self.observation, self.achieved_goal) or truncated:
#             done = True
#         return self.observation, reward, done, self.info
    
#     # def render(self, mode='human'):
#     #     return self.env.render()
    
#     def sample_goal(self):
#         # import pdb;pdb.set_trace
#         return self.env.env._sample_goal()
    
class FetchReachImage(reach.MujocoFetchReachEnv):
  """Wrapper for the FetchReach environment with image observations."""

  def __init__(self, **kwargs):
    self._dist = []
    self._dist_vec = []
    super(FetchReachImage, self).__init__(**kwargs)
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)
    self.observation_space = self._new_observation_space
    self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def reset(self):
    if self._dist:  # if len(self._dist) > 0, ...
      self._dist_vec.append(self._dist)
    self._dist = []

    # generate the new goal image
    self.observation_space = self._old_observation_space
    s = super(FetchReachImage, self).reset()
    self.observation_space = self._new_observation_space
    self._goal = s['desired_goal'].copy()

    for _ in range(10):
      hand = s['achieved_goal']
      obj = s['desired_goal']
      delta = obj - hand
      a = np.concatenate([np.clip(10 * delta, -1, 1), [0.0]])
      s, _, _, _, _ = super(FetchReachImage, self).step(a)

    self._goal_img = self.observation(s)

    self.observation_space = self._old_observation_space
    s = super(FetchReachImage, self).reset()
    self.observation_space = self._new_observation_space
    img = self.observation(s)
    dist = np.linalg.norm(s['achieved_goal'] - self._goal)
    self._dist.append(dist)
    return np.concatenate([img, self._goal_img])

  def step(self, action):
    s, _, _, _, _ = super(FetchReachImage, self).step(action)
    dist = np.linalg.norm(s['achieved_goal'] - self._goal)
    self._dist.append(dist)
    done = False
    r = float(dist < 0.05)
    info = {}
    img = self.observation(s)
    return np.stack([img, self._goal_img]), r, done, info

  def observation(self, observation):
    self.sim.data.site_xpos[0] = 1_000_000
    # img = self.render(mode='rgb_array', height=64, width=64)
    return self.render()

  def _viewer_setup(self):
    super(FetchReachImage, self)._viewer_setup()
    self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.5])
    self.viewer.cam.distance = 0.8
    self.viewer.cam.azimuth = 180
    self.viewer.cam.elevation = -30


class FetchPushImage(push.MujocoPyFetchPushEnv):
  """Wrapper for the FetchPush environment with image observations."""

  def __init__(self, camera='camera2', start_at_obj=True, rand_y=False, **kwargs):
    self._start_at_obj = start_at_obj
    self._rand_y = rand_y
    self._camera_name = camera
    self._dist = []
    self._dist_vec = []
    # super(FetchPushImage, self).__init__()
    super(FetchPushImage, self).__init__(**kwargs)
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)
    self.observation_space = self._new_observation_space
    self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def _move_hand_to_obj(self):
    s = super(FetchPushImage, self)._get_obs()
    for _ in range(100):
      hand = s['observation'][:3]
      obj = s['achieved_goal'] + np.array([-0.02, 0.0, 0.0])
      delta = obj - hand
      if np.linalg.norm(delta) < 0.06:
        break
      a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
      s, _, _, _, _ = super(FetchPushImage, self).step(a)

  def reset(self):
    if self._dist:  # if len(self._dist) > 0 ...
      self._dist_vec.append(self._dist)
    self._dist = []

    # generate the new goal image
    self.observation_space = self._old_observation_space
    s = super(FetchPushImage, self).reset()
    self.observation_space = self._new_observation_space
    # Randomize object position
    for _ in range(8):
      super(FetchPushImage, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    if not self._rand_y:
      object_qpos[1] = 0.75
    self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    self._move_hand_to_obj()
    self._goal_img = self.observation(s)
    block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3]
    if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
      print('Bad reset, recursing.')
      return self.reset()
    self._goal = block_xyz[:2].copy()

    self.observation_space = self._old_observation_space
    s = super(FetchPushImage, self).reset()
    self.observation_space = self._new_observation_space
    for _ in range(8):
      super(FetchPushImage, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    object_qpos[:2] = np.array([1.15, 0.75])
    self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    if self._start_at_obj:
      self._move_hand_to_obj()
    else:
      for _ in range(5):
        super(FetchPushImage, self).step(self.action_space.sample())

    block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
    img = self.observation(s)
    dist = np.linalg.norm(block_xyz[:2] - self._goal)
    self._dist.append(dist)
    if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
      print('Bad reset, recursing.')
      return self.reset()
    return np.stack([img, self._goal_img])

  def step(self, action):
    s, _, _, _, _ = super(FetchPushImage, self).step(action)
    block_xy = self.sim.data.get_joint_qpos('object0:joint')[:2]
    dist = np.linalg.norm(block_xy - self._goal)
    self._dist.append(dist)
    done = False
    r = float(dist < 0.05)  # Taken from the original task code.
    info = {}
    img = self.observation(s)
    return np.concatenate([img, self._goal_img]), r, done, info

  def observation(self, observation):
    self.sim.data.site_xpos[0] = 1_000_000
    # img = self.render(mode='rgb_array', height=64, width=64)
    return self.render()

  def _viewer_setup(self):
    super(FetchPushImage, self)._viewer_setup()
    if self._camera_name == 'camera1':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
      self.viewer.cam.distance = 0.9
      self.viewer.cam.azimuth = 180
      self.viewer.cam.elevation = -40
    elif self._camera_name == 'camera2':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.65
      self.viewer.cam.azimuth = 90
      self.viewer.cam.elevation = -40
    else:
      raise NotImplementedError