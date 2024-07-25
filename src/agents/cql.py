import copy

from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, Critic, ensemblize, DiscretePolicy
from jaxrl_m.evaluation import supply_rng

import flax
import flax.linen as nn
from flax.core import freeze, unfreeze
import ml_collections
from . import iql
from src.special_networks import Representation, HierarchicalActorCritic_HCQL, RelativeRepresentation, MonolithicQF, Scalar

def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)

def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

def compute_actor_loss(agent, batch, network_params):
    # sac policy loss
    # dist = agent.network(batch['observations'], batch['low_goals'], state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
    # new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
    
    # q1, q2 = agent.network(batch['observations'], new_actions, batch['low_goals'], method='qf')
    # q = jnp.minimum(q1, q2)
    
    # alpha = jnp.exp(agent.network(method='log_alpha', params=network_params)) * agent.config['alpha_multiplier']
    # alpha = jnp.clip(alpha, 0, 1)
    # actor_loss = (-q + alpha * log_pi).mean()
    
    # batch['low_log_pi'] = log_pi
    
    # next q
    next_dist = agent.network(batch['observations'], batch['low_goals'], state_rep_grad=True, goal_rep_grad=False, method='actor')
    next_new_actions, next_log_pi = supply_rng(next_dist.sample_and_log_prob)()
    
    next_q1, next_q2 = agent.network(batch['next_observations'], next_new_actions, batch['low_goals'], method='qf')
    next_q = jnp.minimum(next_q1, next_q2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
    
    # cur q    
    dist = agent.network(batch['observations'], batch['low_goals'], state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
    new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
    
    v1, v2 = agent.network(batch['observations'], new_actions, batch['low_goals'], method='qf')
    v = jnp.minimum(v1, v2)
    
    adv = q - v
    exp_a = jnp.clip(jnp.exp(adv), 0, 10)
    mse_loss = jnp.square(new_actions - batch['actions']).mean()
    # target_pi = dist.log_prob(batch['actions'])
    # mse_loss = jnp.square(log_pi - target_pi).mean()
    exp_a = jax.lax.stop_gradient(exp_a)
    actor_loss = (exp_a * mse_loss).mean()

    return actor_loss, {
        'actor_loss': actor_loss,
        # 'alpha' : alpha,
        'exp_a' : exp_a.mean(),
        'q' : q.mean(),
        'v' : v.mean(),
        'adv' : adv.mean(),
        'log_pi' : log_pi.mean(),
        # 'target_pi' : target_pi.mean(),
        'mse': mse_loss.mean(),
        'low_scale': dist.scale_diag.mean(),
        
    }


def compute_high_actor_loss(agent, batch, network_params):
    # sac policy loss
    # dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=False, method='high_actor', params=network_params)
    # new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
    
    # q1, q2 = agent.network(batch['observations'], new_actions, batch['high_goals'], method='high_qf')
    # q = jnp.minimum(q1, q2)
    
    # alpha = jnp.exp(agent.network(method='high_log_alpha', params=network_params)) * agent.config['alpha_multiplier'] 
    # alpha = jnp.clip(alpha, 0, 1)
    
    # actor_loss = (-q + alpha * log_pi).mean()
    
    # batch['high_log_pi'] = log_pi
    
    # next q
    next_dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=False, method='high_actor')
    next_new_actions, next_log_pi = supply_rng(next_dist.sample_and_log_prob)()
    
    next_q1, next_q2 = agent.network(batch['high_targets'], next_new_actions, batch['high_goals'], method='high_qf')
    next_q = jnp.minimum(next_q1, next_q2)
    q = batch['high_rewards'] + agent.config['discount'] * batch['high_masks'] * next_q
    
    # cur q    
    dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=False, method='high_actor', params=network_params)
    new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
    
    v1, v2 = agent.network(batch['observations'], new_actions, batch['high_goals'], method='high_qf')
    v = jnp.minimum(v1, v2)
    
    adv = q - v
    exp_a = jnp.clip(jnp.exp(adv), 0, 10)
    mse_loss = jnp.square(new_actions - batch['high_targets']).mean()
    
    # target_pi = dist.log_prob(batch['high_targets'])
    # mse_loss = jnp.square(log_pi - target_pi).mean()
    exp_a = jax.lax.stop_gradient(exp_a)
    actor_loss = (exp_a * mse_loss).mean()

    return actor_loss, {
        'high_actor_loss': actor_loss,
        # 'alpha' : alpha,
        'exp_a' : exp_a.mean(),
        'q' : q.mean(),
        'v' : v.mean(),
        'adv' : adv.mean(),
        'log_pi' : log_pi.mean(),
        # 'target_pi' : target_pi.mean(),
        'mse': mse_loss.mean(),
        'high_scale': dist.scale_diag.mean(),
    }

def compute_qf_loss(agent, batch, network_params):
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = 1.0 - batch['rewards']
    # rewards are 0 if terminal, -1 otherwise
    batch['rewards'] = batch['rewards'] - 1.0

    # sac-q loss
    next_dist = agent.network(batch['next_observations'], batch['goals'], state_rep_grad=True, goal_rep_grad=False, method='actor')
    next_actions, next_log_pi = supply_rng(next_dist.sample_and_log_prob)(sample_shape=10)
    
    # max target back up
    (next_q1, next_q2) = agent.network(jnp.tile(batch['next_observations'], (10,1,1)), next_actions, jnp.tile(batch['goals'], (10,1,1)), method='target_qf')
    next_q1, next_q2 = jnp.max(next_q1, axis=0), jnp.max(next_q2, axis=0)
    
    next_q = jnp.minimum(next_q1, next_q2)
    
    target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
    target_q = jax.lax.stop_gradient(target_q)

    (q1, q2) = agent.network(batch['observations'], batch['actions'], batch['goals'], method='qf', params=network_params)

    q_loss1 = jnp.square(q1 - target_q).mean()
    q_loss2 = jnp.square(q2 - target_q).mean()


    # cql loss
    batch_size = batch['observations'].shape[0]
    # dimension : (batch_size, cql_n_actions, obs_dim)
    observations =  extend_and_repeat(batch['observations'], 1, agent.config['cql_n_actions'])
    next_observations =  extend_and_repeat(batch['next_observations'], 1, agent.config['cql_n_actions'])
    goals =  extend_and_repeat(batch['goals'], 1, agent.config['cql_n_actions'])
    
    # random actions : (batch_size, cql_n_actions, action_dim)
    cql_random_actions = jax.random.uniform(key=agent.rng, shape=(batch_size, agent.config['cql_n_actions'], agent.config['action_dim']), minval=agent.config['action_min'], maxval=agent.config['action_max'])
    agent = agent.replace(rng = jax.random.split(agent.rng, 1)[0])
    
    cur_dist = agent.network(observations, goals, state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
    cql_current_actions, cql_current_log_pi = supply_rng(cur_dist.sample_and_log_prob)()
    
    next_dist = agent.network(next_observations, goals, state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
    cql_next_actions, cql_next_log_pi = supply_rng(next_dist.sample_and_log_prob)()
    
    
    cql_q1_rand, cql_q2_rand = agent.network(observations, cql_random_actions, goals, method='qf', params=network_params)
    cql_q1_current_actions, cql_q2_current_actions = agent.network(observations, cql_current_actions, goals, method='qf', params=network_params)
    cql_q1_next_actions, cql_q2_next_actions = agent.network(observations, cql_next_actions, goals, method='qf', params=network_params)
    
    # importance sample
    random_density = jnp.log(0.5 ** agent.config['action_dim'])
    cql_cat_q1 = jnp.concatenate([cql_q1_rand - random_density, cql_q1_next_actions - cql_next_log_pi , cql_q1_current_actions - cql_current_log_pi], axis=1)
    cql_cat_q2 = jnp.concatenate([cql_q2_rand - random_density, cql_q2_next_actions - cql_next_log_pi, cql_q2_current_actions - cql_current_log_pi], axis=1)

    cql_qf1_ood = (jax.scipy.special.logsumexp(cql_cat_q1 / agent.config['cql_temp'], axis=1)* agent.config['cql_temp'])
    cql_qf2_ood = (jax.scipy.special.logsumexp(cql_cat_q2 / agent.config['cql_temp'], axis=1)* agent.config['cql_temp'])
    
    cql_qf1_diff = jnp.clip(cql_qf1_ood - q1.mean(), agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
    cql_qf2_diff = jnp.clip(cql_qf2_ood - q2.mean(), agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
                
    # low alpha prime
    log_low_alpha_prime = agent.network(method='log_alpha_prime')
    low_alpha_prime = jnp.clip(jnp.exp(log_low_alpha_prime), 0.0, 1e6)
    low_alpha_prime = jax.lax.stop_gradient(low_alpha_prime)
    
    
    cql_min_qf1_loss = low_alpha_prime * cql_qf1_diff
    cql_min_qf2_loss = low_alpha_prime * cql_qf2_diff
                    
    qf1_loss = q_loss1 + cql_min_qf1_loss
    qf2_loss = q_loss2 + cql_min_qf2_loss
    
    q_loss = qf1_loss + qf2_loss 
    
    # for alpha prime update
    batch['low_cql_qf1_diff'] = cql_qf1_diff
    batch['low_cql_qf2_diff'] = cql_qf2_diff
    
    return q_loss, {
        'q_loss': q_loss,
        'q max': q1.max(),
        'q min': q1.min(),
        'q mean': q1.mean(),
        'q cql_qf_diff': cql_qf1_diff.mean(),
        'q cql_min_qf_loss': cql_min_qf1_loss.mean(),
        'q cql_qf_ood': cql_qf1_ood.mean(),
        'q cql_q_rand': cql_q1_rand.mean(),
        'q cql_q_current_actions': cql_q1_current_actions.mean(),
        'q cql_q_next_actions': cql_q1_next_actions.mean(),
        'low_alpha_prime': low_alpha_prime,

        
    }

def compute_high_qf_loss(agent, batch, network_params):
    # masks are 0 if terminal, 1 otherwise
    batch['high_masks'] = 1.0 - batch['high_rewards']
    # rewards are 0 if terminal, -1 otherwise
    batch['high_rewards'] = batch['high_rewards'] - 1.0
    batch_size, obs_dim = batch['observations'].shape
    

    # sac-q loss
    next_dist = agent.network(batch['high_targets'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=False, method='high_actor')
    next_actions, next_log_pi = supply_rng(next_dist.sample_and_log_prob)(sample_shape=10)
    
    # max target back up
    (next_q1, next_q2) = agent.network(jnp.tile(batch['high_targets'], (10,1,1)), next_actions, jnp.tile(batch['high_goals'], (10,1,1)), method='high_target_qf')
    next_q1, next_q2 = jnp.max(next_q1, axis=0), jnp.max(next_q2, axis=0)
    
    next_q = jnp.minimum(next_q1, next_q2)
    
    target_q = batch['high_rewards'] + agent.config['discount'] * batch['high_masks'] * next_q
    target_q = jax.lax.stop_gradient(target_q)

    (q1, q2) = agent.network(batch['observations'], batch['high_targets'], batch['high_goals'], method='high_qf', params=network_params)

    q_loss1 = jnp.square(q1 - target_q).mean()
    q_loss2 = jnp.square(q2 - target_q).mean()
    

    # cql loss
    # dimension : (batch_size, cql_n_actions, obs_dim)
    observations =  extend_and_repeat(batch['observations'], 1, agent.config['cql_n_actions'])
    high_targets =  extend_and_repeat(batch['high_targets'], 1, agent.config['cql_n_actions'])
    high_goals =  extend_and_repeat(batch['high_goals'], 1, agent.config['cql_n_actions'])
    
    # random actions : (batch_size, cql_n_actions, action_dim)
    cql_random_actions = jax.random.uniform(key=agent.rng, shape=(batch_size, agent.config['cql_n_actions'], obs_dim),minval=agent.config['observation_min'], maxval=agent.config['observation_max'])
    agent = agent.replace(rng=jax.random.split(agent.rng, 1)[0])
    
    cur_dist = agent.network(observations, high_goals, state_rep_grad=True, goal_rep_grad=False, method='high_actor', params=network_params)
    cql_current_actions, cql_current_log_pi = supply_rng(cur_dist.sample_and_log_prob)()
    
    next_dist = agent.network(high_targets, high_goals, state_rep_grad=True, goal_rep_grad=False, method='high_actor', params=network_params)
    cql_next_actions, cql_next_log_pi = supply_rng(next_dist.sample_and_log_prob)()
    
    
    cql_q1_rand, cql_q2_rand = agent.network(observations, cql_random_actions, high_goals, method='high_qf', params=network_params)
    cql_q1_current_actions, cql_q2_current_actions = agent.network(observations, cql_current_actions, high_goals, method='high_qf', params=network_params)
    cql_q1_next_actions, cql_q2_next_actions = agent.network(observations, cql_next_actions, high_goals, method='high_qf', params=network_params)
    
    # importance sample
    random_density = jnp.log(0.5 ** obs_dim)
    cql_cat_q1 = jnp.concatenate([cql_q1_rand - random_density, cql_q1_next_actions - cql_next_log_pi , cql_q1_current_actions - cql_current_log_pi], axis=1)
    cql_cat_q2 = jnp.concatenate([cql_q2_rand - random_density, cql_q2_next_actions - cql_next_log_pi, cql_q2_current_actions - cql_current_log_pi], axis=1)

    cql_qf1_ood = (jax.scipy.special.logsumexp(cql_cat_q1 / agent.config['cql_temp'], axis=1)* agent.config['cql_temp'])
    cql_qf2_ood = (jax.scipy.special.logsumexp(cql_cat_q2 / agent.config['cql_temp'], axis=1)* agent.config['cql_temp'])
    
    cql_qf1_diff = jnp.clip(cql_qf1_ood - q1.mean(), agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
    cql_qf2_diff = jnp.clip(cql_qf2_ood - q2.mean(), agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
    
    # high alpha prime
    log_high_alpha_prime = agent.network(method='high_log_alpha_prime')
    high_alpha_prime = jnp.clip(jnp.exp(log_high_alpha_prime), 0.0, 1e6)
    
    high_alpha_prime = jax.lax.stop_gradient(high_alpha_prime)
    
    cql_min_qf1_loss = high_alpha_prime * cql_qf1_diff
    cql_min_qf2_loss = high_alpha_prime * cql_qf2_diff
                    
    qf1_loss = q_loss1 + cql_min_qf1_loss
    qf2_loss = q_loss2 + cql_min_qf2_loss
    
    # q_loss = qf1_loss + qf2_loss + high_alpha_prime_loss
    q_loss = qf1_loss + qf2_loss 
    
    # for alpha prime update
    batch['high_cql_qf2_diff'] = cql_qf2_diff
    batch['high_cql_qf1_diff'] = cql_qf1_diff
    
    
    # q_loss = q_loss1 + q_loss2 
    
    
    return q_loss, {
        'q_loss': q_loss,
        'q max': q1.max(),
        'q min': q1.min(),
        'q mean': q1.mean(),
        'q cql_qf_diff': cql_qf1_diff.mean(),
        'q cql_min_qf_loss': cql_min_qf1_loss.mean(),
        'q cql_qf_ood': cql_qf1_ood.mean(),
        'q cql_q_rand': cql_q1_rand.mean(),
        'q cql_q_current_actions': cql_q1_current_actions.mean(),
        'q cql_q_next_actions': cql_q1_next_actions.mean(),
        'high_alpha_prime': high_alpha_prime,
    }
    
def compute_low_alpha_loss(agent, batch, network_params):
    # low log alpha
    dist = agent.network(batch['observations'], batch['goals'], state_rep_grad=True, goal_rep_grad=False, method='actor')
    actions, low_log_pi = supply_rng(dist.sample_and_log_prob)()
    # low_log_pi = jax.lax.stop_gradient(batch['low_log_pi'])
    
    log_alpha = agent.network(method='log_alpha', params=network_params)
    # low_alpha_loss = -log_alpha * (low_log_pi + agent.config['target_entropy']).mean()
    low_alpha_loss = -log_alpha * (low_log_pi).mean()
    
    return low_alpha_loss, {
        'low_alpha_loss': low_alpha_loss,
        'alpha' : jnp.exp(log_alpha),
        }

def compute_high_alpha_loss(agent, batch, network_params):
    # high log alpha
    high_dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=False, method='high_actor')
    actions, high_log_pi = supply_rng(high_dist.sample_and_log_prob)()
    # high_log_pi = jax.lax.stop_gradient(batch['high_log_pi'])
    
    log_high_alpha = agent.network(method='high_log_alpha', params=network_params)
    # high_alpha_loss = -log_high_alpha * (high_log_pi + agent.config['high_target_entropy']).mean()
    high_alpha_loss = -log_high_alpha * (high_log_pi).mean()
    
    return high_alpha_loss, {
        'high_alpha_loss': high_alpha_loss,
        'high_alpha' : jnp.exp(log_high_alpha),
        }
    
def compute_low_alpha_prime_loss(agent, batch, network_params):
    low_cql_qf1_diff = jax.lax.stop_gradient(batch['low_cql_qf1_diff'])
    low_cql_qf2_diff = jax.lax.stop_gradient(batch['low_cql_qf2_diff'])
    
    # high alpha prime
    log_alpha_prime = agent.network(method='log_alpha_prime', params=network_params)
    
    cql_min_qf1_loss = (low_cql_qf1_diff - agent.config['cql_low_target_action_gap']) * agent.config['cql_min_q_weight']
    cql_min_qf2_loss = (low_cql_qf2_diff - agent.config['cql_low_target_action_gap']) * agent.config['cql_min_q_weight']
                    
    low_alpha_prime_loss = - log_alpha_prime * (cql_min_qf1_loss + cql_min_qf2_loss)*0.5
    
    return low_alpha_prime_loss , {
        'low_alpha_prime' : jnp.exp(log_alpha_prime)
    }

def compute_high_alpha_prime_loss(agent, batch, network_params):
    high_cql_qf1_diff = jax.lax.stop_gradient(batch['high_cql_qf1_diff'])
    high_cql_qf2_diff = jax.lax.stop_gradient(batch['high_cql_qf2_diff'])
    
    # high alpha prime
    log_high_alpha_prime = agent.network(method='high_log_alpha_prime', params=network_params)
    
    cql_min_qf1_loss = (high_cql_qf1_diff - agent.config['cql_high_target_action_gap']) * agent.config['cql_min_q_weight']
    cql_min_qf2_loss =  (high_cql_qf2_diff - agent.config['cql_high_target_action_gap']) * agent.config['cql_min_q_weight']
                    
    high_alpha_prime_loss = - log_high_alpha_prime * (cql_min_qf1_loss + cql_min_qf2_loss)*0.5
    
    return high_alpha_prime_loss , {
        'high_alpha_prime' : jnp.exp(log_high_alpha_prime)
    }

class JointTrainAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    qf: TrainState
    target_qf: TrainState
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)
    network: TrainState = None
    key_nodes: dict = None
    
    def pretrain_update(agent, pretrain_batch, seed=None, qf_update=True, actor_update=True, alpha_update=True, high_actor_update=True,  hilp_update=True):
        def loss_fn(network_params):
            info = {}

            # Q function
            if qf_update:
                qf_loss, qf_info = compute_qf_loss(agent, pretrain_batch, network_params)
                for k, v in qf_info.items():
                    info[f'qf/{k}'] = v
            else:
                qf_loss = 0.

            # high Q function
            if qf_update:
                high_qf_loss, high_qf_info = compute_high_qf_loss(agent, pretrain_batch, network_params)
                for k, v in high_qf_info.items():
                    info[f'high_qf/{k}'] = v
            else:
                high_qf_loss = 0.
                
            # Actor
            if actor_update:
                actor_loss, actor_info = compute_actor_loss(agent, pretrain_batch, network_params)
                for k, v in actor_info.items():
                    info[f'actor/{k}'] = v
            else:
                actor_loss = 0.

            # High Actor
            if high_actor_update:
                high_actor_loss, high_actor_info = compute_high_actor_loss(agent, pretrain_batch, network_params)
                for k, v in high_actor_info.items():
                    info[f'high_actor/{k}'] = v
            else:
                high_actor_loss = 0.

            loss = qf_loss + high_qf_loss + actor_loss + high_actor_loss

            return loss, info

        def low_alpha_loss_fn(network_params):
            info = {}
               
            # alpha function
            low_alpha_loss, alpha_info = compute_low_alpha_loss(agent, pretrain_batch, network_params)
            for k, v in alpha_info.items():
                info[f'alpha/{k}'] = v

            return low_alpha_loss, info

        def high_alpha_loss_fn(network_params):
            info = {}
               
            # alpha function
            high_alpha_loss, alpha_info = compute_high_alpha_loss(agent, pretrain_batch, network_params)
            for k, v in alpha_info.items():
                info[f'alpha/{k}'] = v

            return high_alpha_loss, info

        def low_alpha_prime_loss_fn(network_params):
            info = {}
               
            # alpha function
            low_alpha_prime_loss, alpha_prime_info = compute_low_alpha_prime_loss(agent, pretrain_batch, network_params)
            for k, v in alpha_prime_info.items():
                info[f'alpha/{k}'] = v

            return low_alpha_prime_loss, info

        def high_alpha_prime_loss_fn(network_params):
            info = {}
               
            # alpha function
            high_alpha_prime_loss, alpha_prime_info = compute_high_alpha_prime_loss(agent, pretrain_batch, network_params)
            for k, v in alpha_prime_info.items():
                info[f'alpha/{k}'] = v

            return high_alpha_prime_loss, info
        
        if qf_update:
            new_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_qf'], agent.network.params['networks_target_qf']
            )
            new_high_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_high_qf'], agent.network.params['networks_high_target_qf']
            )
        # Q fn, policy update
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        agent = agent.replace(network=new_network)
        
        if qf_update:
            params = unfreeze(new_network.params)
            params['networks_target_qf'] = new_target_params
            params['networks_high_target_qf'] = new_high_target_params
            new_network = new_network.replace(params=freeze(params))
        
        # additional update
        # new_params = unfreeze(agent.network.params)
        
        # # low alpha update
        # network, low_alpha_info = agent.network.apply_loss_fn(loss_fn=low_alpha_loss_fn, has_aux=True)
        # info.update(low_alpha_info)
        # new_params['networks_log_alpha'] = network.params['networks_log_alpha']
        
        # # high alpha update
        # network, high_alpha_info = agent.network.apply_loss_fn(loss_fn=high_alpha_loss_fn, has_aux=True)
        # info.update(high_alpha_info)
        # new_params['networks_high_log_alpha'] = network.params['networks_high_log_alpha']
        
        # # low alpha prime
        # network, low_alpha_prime_info = agent.network.apply_loss_fn(loss_fn=low_alpha_prime_loss_fn, has_aux=True)
        # info.update(low_alpha_prime_info)        
        # new_params['networks_log_alpha_prime'] = network.params['networks_log_alpha_prime']
        
        # # high alpha prime
        # network, high_alpha_prime_info = agent.network.apply_loss_fn(loss_fn=high_alpha_prime_loss_fn, has_aux=True)
        # info.update(high_alpha_prime_info)
        # new_params['networks_high_log_alpha_prime'] = network.params['networks_high_log_alpha_prime']
        
        # new_network = new_network.replace(params=freeze(new_params))
        
        return agent.replace(network=new_network), info
        # return agent.replace(params=freeze(new_params)), info
    pretrain_update = jax.jit(pretrain_update, static_argnames=('qf_update', 'actor_update', 'alpha_update', 'high_actor_update'))

    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       *,
                       low_dim_goals: bool = False,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, low_dim_goals=low_dim_goals, temperature=temperature, method='actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        if not discrete:
            actions = jnp.clip(actions, -1, 1)
        return actions
    sample_actions = jax.jit(sample_actions, static_argnames=('num_samples', 'low_dim_goals', 'discrete'))

    def sample_high_actions(agent,
                            observations: np.ndarray,
                            goals: np.ndarray,
                            *,
                            seed: PRNGKey,
                            temperature: float = 1.0,
                            num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, temperature=temperature, method='high_actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        return actions
    sample_high_actions = jax.jit(sample_high_actions, static_argnames=('num_samples',))

    @jax.jit
    def get_policy_rep(agent,
                       *,
                       targets: np.ndarray,
                       bases: np.ndarray = None,
                       ) -> jnp.ndarray:
        return agent.network(targets=targets, bases=bases, method='policy_goal_encoder')


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        # goals: jnp.ndarray,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        qf_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        temperature: float = 1,
        high_temperature: float = 1,
        way_steps: int = 0,
        rep_dim: int = 10,
        use_rep: int = 0,
        policy_train_rep: float = 0,
        visual: int = 0,
        encoder: str = 'impala',
        discrete: int = 0,
        use_layer_norm: int = 0,
        # rep_type: str = 'state',
        use_waypoints: int = 0,
        key_nodes : Any = None,
        flag: Any = None,
        use_automatic_entropy_tuning = True,
        backup_entropy = False,
        policy_lr = 1e-4,
        qf_lr = 3e-4,
        optimizer_type = 'adam',
        soft_target_update_rate = 5e-3,
        cql_n_actions = 10,
        cql_importance_sample = True,
        cql_lagrange = False,
        cql_target_action_gap = 1.0,
        cql_temp = 1.0,
        cql_min_q_weight= 5.0, 
        cql_max_target_backup = True,
        cql_clip_diff_min = -np.inf,
        cql_clip_diff_max = np.inf,
        high_alpha_multiplier = 1,
        alpha_multiplier = 1,
        cql_low_target_action_gap = 1,
        cql_high_target_action_gap = 1,        
        **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, high_actor_key, critic_key, qf_key = jax.random.split(rng, 5)

        qf_state_encoder = None
        qf_goal_encoder = None
        high_qf_state_encoder = None
        high_qf_goal_encoder = None
        policy_state_encoder = None
        policy_goal_encoder = None
        high_policy_state_encoder = None
        high_policy_goal_encoder = None
        log_alpha = None
        
        if visual:
            assert use_rep
            from jaxrl_m.vision import encoders

            visual_encoder = encoders[encoder]
            def make_encoder(bottleneck):
                if bottleneck:
                    return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(rep_dim,), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=flag.rep_type, bottleneck=True)
                else:
                    return RelativeRepresentation(rep_dim=qf_hidden_dims[-1], hidden_dims=(qf_hidden_dims[-1],), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=flag.rep_type, bottleneck=False)

            qf_state_encoder = make_encoder(bottleneck=False)
            qf_goal_encoder = make_encoder(bottleneck=use_waypoints)
            high_qf_state_encoder = make_encoder(bottleneck=False)
            high_qf_goal_encoder = make_encoder(bottleneck=use_waypoints)
            policy_state_encoder = make_encoder(bottleneck=False)
            policy_goal_encoder = make_encoder(bottleneck=False)
            high_policy_state_encoder = make_encoder(bottleneck=False)
            high_policy_goal_encoder = make_encoder(bottleneck=False)
        else:
            def make_encoder(bottleneck):
                if bottleneck:
                    return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(*qf_hidden_dims, rep_dim), layer_norm=use_layer_norm, rep_type=flag.rep_type, bottleneck=True)
                else:
                    return RelativeRepresentation(rep_dim=qf_hidden_dims[-1], hidden_dims=(*qf_hidden_dims, qf_hidden_dims[-1]), layer_norm=use_layer_norm, rep_type=flag.rep_type, bottleneck=False)

            if use_rep:
                qf_goal_encoder = make_encoder(bottleneck=True)

        qf_def = MonolithicQF(hidden_dims=qf_hidden_dims, use_layer_norm=use_layer_norm)
        high_qf_def = MonolithicQF(hidden_dims=qf_hidden_dims, use_layer_norm=use_layer_norm)
        log_alpha = Scalar()
        high_log_alpha = Scalar()
        log_alpha_prime = Scalar()
        high_log_alpha_prime = Scalar()
        
        if discrete:
            action_dim = actions[0] + 1
            actor_def = DiscretePolicy(actor_hidden_dims, action_dim=action_dim)
        else:
            action_dim = actions.shape[-1]
            actor_def = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        high_action_dim = observations.shape[-1] if not use_rep else rep_dim
        high_actor_def = Policy(actor_hidden_dims, action_dim=high_action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        network_def = HierarchicalActorCritic_HCQL(
            encoders={
                'qf_state': qf_state_encoder,
                'qf_goal': qf_goal_encoder,
                'high_qf_state': high_qf_state_encoder,
                'high_qf_goal': high_qf_goal_encoder,
                'policy_state': policy_state_encoder,
                'policy_goal': policy_goal_encoder,
                'high_policy_state': high_policy_state_encoder,
                'high_policy_goal': high_policy_goal_encoder,
            },
            networks={
                'qf': qf_def,
                'target_qf': copy.deepcopy(qf_def),
                'high_qf': high_qf_def,
                'high_target_qf': copy.deepcopy(high_qf_def),
                'actor': actor_def,
                'high_actor': high_actor_def,
                'log_alpha' : log_alpha,
                'high_log_alpha' : high_log_alpha,
                'log_alpha_prime' : log_alpha_prime,
                'high_log_alpha_prime' : high_log_alpha_prime,
            },
            flag=flag,
        )
        network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
        network_params = network_def.init(qf_key, observations, actions, observations)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = unfreeze(network.params)
        params['networks_target_qf'] = params['networks_qf']
        network = network.replace(params=freeze(params))

        # config = flax.core.FrozenDict(dict(
        #     discount=discount, temperature=temperature, high_temperature=high_temperature,
        #     target_update_rate=tau, way_steps=way_steps, rep_dim=rep_dim,
        #     policy_train_rep=policy_train_rep,
        #     use_rep=use_rep, use_waypoints=use_waypoints,
        # ))
        flag_dict = flag.flag_values_dict()
        flag_dict.update(kwargs)
        
        
        config = flax.core.FrozenDict(**flag_dict,  **{'target_update_rate':tau, 'cql_n_actions':cql_n_actions, 'alpha_multiplier' : alpha_multiplier, 'use_automatic_entropy_tuning' : use_automatic_entropy_tuning, 'backup_entropy' : backup_entropy, 'policy_lr' : policy_lr, 'cql_min_q_weight' :cql_min_q_weight, 'qf_lr' : qf_lr, 'optimizer_type' : optimizer_type, 'soft_target_update_rate' : soft_target_update_rate, 'cql_n_actions' : cql_n_actions, 'cql_importance_sample' : cql_importance_sample, 'cql_lagrange' : cql_lagrange, 'cql_target_action_gap' : cql_target_action_gap, 'cql_temp' : cql_temp, 'cql_max_target_backup' : cql_max_target_backup, 'cql_clip_diff_min' : cql_clip_diff_min, 'cql_clip_diff_max' : cql_clip_diff_max, 'action_dim':action_dim, 'high_action_dim':high_action_dim, 'high_alpha_multiplier':high_alpha_multiplier, 'alpha_multiplier':alpha_multiplier, 'cql_low_target_action_gap':cql_low_target_action_gap, 'cql_high_target_action_gap': cql_high_target_action_gap})

        return JointTrainAgent(rng, network=network, qf=None, target_qf=None, actor=None, config=config, key_nodes=key_nodes)


def get_default_config():
    config = ml_collections.ConfigDict({
        'lr': 3e-4,
        'actor_hidden_dims': (256, 256),
        'qf_hidden_dims': (256, 256),
        'discount': 0.99,
        'temperature': 1.0,
        'tau': 0.005,
        'pretrain_expectile': 0.7,
    })

    return config