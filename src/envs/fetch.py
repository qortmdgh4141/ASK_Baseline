from gym.core import Wrapper
from jaxrl_m.dataset import Dataset
import numpy as np 

def fetch_load(env_name, data):

    observations = data['o'][:,:-1].reshape(-1, data['o'].shape[-1])
    next_observations = data['o'][:,1:].reshape(-1, data['o'].shape[-1])
    dones_float = np.zeros((data['o'].shape[0],data['o'].shape[1]-1))
    dones_float[:,:-1] = 1
    dones_float = dones_float.reshape(-1)
    
    return Dataset.create(
            observations=observations.astype(np.float32),
            actions=data['u'].reshape(-1, data['u'].shape[-1]).astype(np.float32),
            rewards=None,
            masks=None,
            dones_float=dones_float,
            next_observations=next_observations.astype(np.float32),
            returns = np.ones_like(dones_float).astype(np.float32),
            goal_info = data['g'].reshape(-1, data['g'].shape[-1]),
            ), None

def reach_goal(state, desired_goal):
    epsilon = 0.05
    achieved_goal = state[:desired_goal.shape[-1]]

    return np.linalg.norm((achieved_goal, desired_goal)) < epsilon

class FetchGoalWrapper(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    def reset(self):
        return self.env.reset()
    
    def compute_rewards(self, achieved_goal, desired_goal, info=None):
        return self.env.compute_reward(self.achieved_goal, self.desired_goal, self.info)

    def step(self, action):
        state, reward, done, truncated, self.info = self.env.step(action)
        # Fetch default spase reward : success = 0, fail = -1 
        # reward + 1 --> success = 1, fail = 0
        reward +=1 
        self.observation, self.achieved_goal, self.desired_goal = state['observation'], state['achieved_goal'], state['desired_goal']
        
        if reach_goal(self.observation, self.achieved_goal) or truncated:
            done = True
        return self.observation, reward, done, self.info
    
    # def render(self, mode='human'):
    #     return self.env.render()
    
    def sample_goal(self):
        # import pdb;pdb.set_trace
        return self.env.env._sample_goal()