from jaxrl_m.dataset import Dataset
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import ml_collections
from typing import *

@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    geom_sample: int
    discount: float
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    terminal: bool = True
    use_rep: str = ""
    # find_key_node_in_dataset : Callable = None
    key_node: Any = None
    
    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
        })

    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)
        # if self.find_key_node_in_dataset is not None:
        #     if 'key_node' in self.dataset.keys():
        #         self.key_node = self.find_key_node_in_dataset(self.dataset['rep_observations'])
            # else:
            #     # self.key_node = self.find_key_node(self.dataset['observations'])
            #     raise NotImplementedError

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal
            
        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        
        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)
        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)
        
        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)
        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])
        return batch

@dataclasses.dataclass
class GCSDataset(GCDataset):
    way_steps: int = None
    high_p_randomgoal: float = 0.
    keynode_ratio: float = 0.5
    high_p_relable : float = 0.
    # find_key_node_in_dataset : Any = None
    
    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': 0.0,
            'terminal': False,
        })

    def sample(self, batch_size: int, indx=None, hilp=False):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        # goal for value function training
        goal_indx = self.sample_goals(indx)
        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift

        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)


        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])

        # final state in each trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        # subgoal sampled from its own trajectory for low level training
        way_indx = np.minimum(indx + self.way_steps, final_state_indx)
        batch['low_goals'] = jax.tree_map(lambda arr: arr[way_indx], self.dataset['observations'])
        
        if hilp == False:
            # subgoal sampled from its own trajectory with distance ratio for high level training
            distance = np.random.rand(batch_size)
            high_traj_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)
            # subgoal sampled from its own trajectory for high level training 
            high_traj_target_indx = np.minimum(indx + self.way_steps, high_traj_goal_indx)
            # randaom goal sampled from 
            high_random_goal_indx = np.random.randint(self.dataset.size, size=batch_size)
            high_random_target_indx = np.minimum(indx + self.way_steps, final_state_indx)

            pick_random = (np.random.rand(batch_size) < self.high_p_randomgoal)
            high_goal_idx = np.where(pick_random, high_random_goal_indx, high_traj_goal_indx)
            high_target_idx = np.where(pick_random, high_random_target_indx, high_traj_target_indx)
            
            batch['high_goals'] = jax.tree_map(lambda arr: arr[high_goal_idx], self.dataset['observations'])
            
            batch['high_targets'] = jax.tree_map(lambda arr: arr[high_target_idx], self.dataset['observations'])
            
            # goal_indx = self.sample_goals(indx)
            # high_success = (high_target_idx == high_goal_idx)
            
            # additional relable for high level
            relable = (np.random.rand(batch_size) < self.high_p_relable)
            high_goal_idx = np.where(relable, high_traj_target_indx, high_random_target_indx)
            
            high_success = (high_target_idx == high_goal_idx)
            
            batch['high_rewards'] = high_success.astype(float) * self.reward_scale + self.reward_shift
            batch['high_masks'] = (1.0 - high_success.astype(float))
            
            
            if 'key_node' in self.dataset.keys():
                batch['key_node'] = self.dataset['key_node'][indx]
                if 'rep_observations' in self.dataset.keys():
                    batch['rep_low_goals'] = jax.tree_map(lambda arr: arr[way_indx], self.dataset['rep_observations'])
                

                # if self.keynode_ratio:
                #     index =  int(self.keynode_ratio* batch_size)
                #     way_index_key_node = way_indx[:index]
                #     # way_index_obs = way_indx[index:]
                #     low_goals_key_node = jax.tree_map(lambda arr: arr[way_index_key_node], self.key_node)
                #     low_goals_obs = jax.tree_map(lambda arr: arr[way_index_obs], self.dataset['observations'])
                #     batch['low_goals'] = jnp.concatenate([low_goals_key_node, low_goals_obs])
                    
                # else:
                #     batch['low_goals'] = jax.tree_map(lambda arr: arr[way_indx], self.dataset['observations'])
                    
                # if self.keynode_ratio:
                #     index = int(self.keynode_ratio* batch_size)
                #     high_target_index_key_node = high_target_idx[:index]
                #     high_target_index_obs = high_target_idx[index:]
                #     high_targets_key_node = jax.tree_map(lambda arr: arr[high_target_index_key_node], self.key_node)
                #     high_targets_obs = jax.tree_map(lambda arr: arr[high_target_index_obs], self.dataset['observations'])
                #     batch['high_targets'] = jnp.concatenate([high_targets_key_node, high_targets_obs])
                # else:
                #     batch['high_targets'] = jax.tree_map(lambda arr: arr[high_target_idx], self.dataset['observations'])

        if isinstance(batch['goals'], FrozenDict):
            # Freeze the other observations
            batch['observations'] = freeze(batch['observations'])
            batch['next_observations'] = freeze(batch['next_observations'])
 
        return batch
