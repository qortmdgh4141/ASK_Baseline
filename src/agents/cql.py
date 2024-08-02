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
from src.special_networks import Representation, HierarchicalActorCritic_HCQL, RelativeRepresentation, MonolithicQF, Scalar, HILP_GoalConditionedPhiValue
from src.utils import get_encoder

def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)

def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

def compute_kl(post_mean, post_std, prior_mean, prior_std=1):
    kl = jnp.log(prior_std) - jnp.log(post_std) + 0.5 * ((post_std**2 + (post_mean - prior_mean)**2) / prior_std**2 - 1)
    return jnp.mean(kl, axis=-1)

def compute_actor_loss(agent, batch, network_params):
    # sac policy loss
    # dist = agent.network(batch['observations'], batch['low_goals'], state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
    # new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
    
    # q1, q2 = agent.network(batch['observations'], new_actions, batch['low_goals'], method='qf')
    # q = jnp.minimum(q1, q2)
    
    # alpha = jnp.exp(agent.network(method='log_alpha')) * agent.config['alpha_multiplier']
    # alpha = jnp.clip(alpha, 0, 1)
    # actor_loss = (-q + alpha * log_pi).mean()
    
    # batch['low_log_pi'] = log_pi
    
    if agent.config['high_action_in_hilp']:
        # subgoals = (batch['rep_low_goals'] - batch['rep_observations']) / jnp.linalg.norm(batch['rep_low_goals'] - batch['rep_observations'], axis=-1, keepdims=True)
        subgoals = jnp.concatenate([batch['rep_observations'], batch['rep_low_goals']], axis=1)
        
    else:
        subgoals = batch['low_goals']
        
    
    # next q
    if agent.config['final_goal']:
        next_dist = agent.network(batch['next_observations'], subgoals, batch['final_goals'], state_rep_grad=True, goal_rep_grad=False, method='actor')
        next_new_actions, next_log_pi = supply_rng(next_dist.sample_and_log_prob)()
        
        next_q1, next_q2 = agent.network(batch['next_observations'], next_new_actions, subgoals, batch['final_goals'], method='qf')
        next_q = jnp.minimum(next_q1, next_q2)
        q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
        
        # cur q    
        dist = agent.network(batch['observations'], subgoals, batch['final_goals'],state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
        new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
        
        v1, v2 = agent.network(batch['observations'], new_actions, subgoals, batch['final_goals'], method='qf')
    else:
        next_dist = agent.network(batch['next_observations'], subgoals, state_rep_grad=True, goal_rep_grad=False, method='actor')
        next_new_actions, next_log_pi = supply_rng(next_dist.sample_and_log_prob)()
        
        next_q1, next_q2 = agent.network(batch['next_observations'], next_new_actions, subgoals, method='qf')
        next_q = jnp.minimum(next_q1, next_q2)
        q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
        
        # cur q    
        dist = agent.network(batch['observations'], subgoals, state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
        new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
        
        v1, v2 = agent.network(batch['observations'], new_actions, subgoals, method='qf')
        
    v = jnp.minimum(v1, v2)
    
    adv = q - v
    exp_a = jnp.clip(jnp.exp(adv), 0, 100)
    exp_a = jax.lax.stop_gradient(exp_a)
    # mse loss
    mse_loss = jnp.square(new_actions - batch['actions']).mean()
    actor_loss = (exp_a * mse_loss).mean()
    
    # guider style - kl loss
    # kl_loss = compute_kl(dist.loc, dist.scale_diag, batch['actions'])
    # actor_loss = (-v + kl_loss).mean() 
    
    # sac alpha = 1
    # actor_loss = (-v + log_pi).mean() 
    


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
        # 'kl_loss': kl_loss.mean(),
        'low_scale': dist.scale_diag.mean(),
        
    }


def compute_high_actor_loss(agent, batch, network_params):
    # sac policy loss
    # dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=False, method='high_actor', params=network_params)
    # new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
    
    # q1, q2 = agent.network(batch['observations'], new_actions, batch['high_goals'], method='high_qf')
    # q = jnp.minimum(q1, q2)
    
    alpha = jnp.exp(agent.network(method='high_log_alpha')) * agent.config['alpha_multiplier'] 
    alpha = jnp.clip(alpha, 0, 1)
    
    # actor_loss = (-q + alpha*log_pi).mean()
    
    # batch['high_log_pi'] = log_pi
    
    # next q
    # next_dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=False, method='high_actor')
    # next_new_actions, next_log_pi = supply_rng(next_dist.sample_and_log_prob)()
    
    # next_q1, next_q2 = agent.network(batch['high_targets'], next_new_actions, batch['high_goals'], method='high_qf')
    # next_q = jnp.minimum(next_q1, next_q2)
    # q = batch['high_rewards'] + agent.config['discount'] * batch['high_masks'] * next_q
    
    # cur q    
    dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=False, method='high_actor', params=network_params)
    new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
    
    v1, v2 = agent.network(batch['observations'], new_actions, batch['high_goals'], method='high_qf')
    v = jnp.minimum(v1, v2)
    
    # adv = q - v
    # exp_a = jnp.clip(jnp.exp(adv), 0, 100)
    # exp_a = jax.lax.stop_gradient(exp_a)
    
    # mse loss
    if agent.config['high_action_in_hilp']:
        target = batch['rep_high_targets']
        # target = jnp.concatenate([batch['rep_observations'], batch['rep_low_goals']], axis=-1)
        # target = (batch['rep_low_goals'] - batch['rep_observations']) / jnp.linalg.norm(batch['rep_low_goals'] - batch['rep_observations'], axis=-1, keepdims=True)
    elif agent.config['key_node_train']:
        target = batch['high_target_key_node']
    else:
        target = batch['high_targets']
    # mse_loss = jnp.square(new_actions - target).mean()
    # actor_loss = (exp_a * mse_loss).mean()
    # v = jax.lax.stop_gradient(v)
    # actor_loss = (-v + alpha*mse_loss).mean()

    kl_loss = compute_kl(dist.loc, dist.scale_diag, target, prior_std=1)
    # compute_kl(post_mean, post_std, prior_mean, prior_std=1)
    actor_loss = (-v + alpha*kl_loss).mean()
    
    batch['high_kl_loss'] = kl_loss

    
    # guider style - kl loss
    # if agent.config['key_node_q']:
    #     targets = batch['high_target_key_node']
    # else: 
    #     targets = batch['high_targets']
    # kl_loss = compute_kl(dist.loc, dist.scale_diag, targets)
    # actor_loss = (-v + kl_loss).mean() 
    
    # sac alpha = 1
    # actor_loss = (-v + log_pi).mean() 


    return actor_loss, {
        'high_actor_loss': actor_loss,
        # 'alpha' : alpha,
        # 'exp_a' : exp_a.mean(),
        # 'q' : q.mean(),
        'v' : v.mean(),
        # 'adv' : adv.mean(),
        'log_pi' : log_pi.mean(),
        # 'target_pi' : target_pi.mean(),
        # 'mse': mse_loss.mean(),
        'kl_loss': kl_loss.mean(),
        'high_scale': dist.scale_diag.mean(),
    }

def compute_qf_loss(agent, batch, network_params):
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = 1.0 - batch['rewards']
    # rewards are 0 if terminal, -1 otherwise
    batch['rewards'] = batch['rewards'] - 1.0

    # sac-q loss
    # max target back up
    if agent.config['high_action_in_hilp']:
        # subgoals = batch['rep_low_goals'] - batch['rep_observations']
        # subgoals = (batch['rep_low_goals'] - batch['rep_observations']) / jnp.linalg.norm(batch['rep_low_goals'] - batch['rep_observations'], axis=-1, keepdims=True)
        subgoals = jnp.concatenate([batch['rep_observations'], batch['rep_low_goals']], axis=1)
        
        
    else:
        subgoals = batch['goals']
    
    if agent.config['final_goal']:
        next_dist = agent.network(batch['next_observations'], subgoals, batch['final_goals'], state_rep_grad=True, goal_rep_grad=False, method='actor')
        next_actions, next_log_pi = supply_rng(next_dist.sample_and_log_prob)(sample_shape=10)
        
        (next_q1, next_q2) = agent.network(jnp.tile(batch['next_observations'], (10,1,1)), next_actions, jnp.tile(subgoals, (10,1,1)), jnp.tile(batch['final_goals'], (10,1,1)), method='target_qf')
        next_q_ = jnp.minimum(next_q1, next_q2)
        
        max_q_index = jnp.argmax(next_q_, axis=0)
        next_q_max = jnp.take_along_axis(next_q_.transpose(1,0), jnp.expand_dims(max_q_index, axis=-1), axis=-1) 
        next_log_pi_max = jnp.take_along_axis(next_log_pi.transpose(1,0), jnp.expand_dims(max_q_index, axis=-1), axis=-1) 
        
        # entropy 
        # log_alpha = agent.network(method='log_alpha')
        # alpha = jnp.exp(log_alpha)
        # next_q = next_q_max - alpha * next_log_pi_max
        
        # no entropy
        next_q = next_q_max
        
        target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        (q1, q2) = agent.network(batch['observations'], batch['actions'], subgoals, batch['final_goals'], method='qf', params=network_params)
    
    else:
        next_dist = agent.network(batch['next_observations'], subgoals, state_rep_grad=True, goal_rep_grad=False, method='actor')
        next_actions, next_log_pi = supply_rng(next_dist.sample_and_log_prob)(sample_shape=10)
        
        (next_q1, next_q2) = agent.network(jnp.tile(batch['next_observations'], (10,1,1)), next_actions, jnp.tile(subgoals, (10,1,1)), method='target_qf')
        next_q_ = jnp.minimum(next_q1, next_q2)
        
        max_q_index = jnp.argmax(next_q_, axis=0)
        next_q_max = jnp.take_along_axis(next_q_.transpose(1,0), jnp.expand_dims(max_q_index, axis=-1), axis=-1) 
        next_log_pi_max = jnp.take_along_axis(next_log_pi.transpose(1,0), jnp.expand_dims(max_q_index, axis=-1), axis=-1) 
        
        # entropy 
        # log_alpha = agent.network(method='log_alpha')
        # alpha = jnp.exp(log_alpha)
        # next_q = next_q_max - alpha * next_log_pi_max
        
        # no entropy
        next_q = next_q_max
        
        target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        (q1, q2) = agent.network(batch['observations'], batch['actions'], subgoals, method='qf', params=network_params)

    q_loss1 = jnp.square(q1 - target_q).mean()
    q_loss2 = jnp.square(q2 - target_q).mean()
    q_loss = q_loss1 + q_loss2

    # cql loss
    # batch_size = batch['observations'].shape[0]
    # # dimension : (batch_size, cql_n_actions, obs_dim)
    # observations =  extend_and_repeat(batch['observations'], 1, agent.config['cql_n_actions'])
    # next_observations =  extend_and_repeat(batch['next_observations'], 1, agent.config['cql_n_actions'])
    # goals =  extend_and_repeat(batch['goals'], 1, agent.config['cql_n_actions'])
    
    # # random actions : (batch_size, cql_n_actions, action_dim)
    # cql_random_actions = jax.random.uniform(key=agent.rng, shape=(batch_size, agent.config['cql_n_actions'], agent.config['action_dim']), minval=agent.config['action_min'], maxval=agent.config['action_max'])
    # agent = agent.replace(rng = jax.random.split(agent.rng, 1)[0])
    
    # cur_dist = agent.network(observations, goals, state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
    # cql_current_actions, cql_current_log_pi = supply_rng(cur_dist.sample_and_log_prob)()
    
    # next_dist = agent.network(next_observations, goals, state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
    # cql_next_actions, cql_next_log_pi = supply_rng(next_dist.sample_and_log_prob)()
    
    
    # cql_q1_rand, cql_q2_rand = agent.network(observations, cql_random_actions, goals, method='qf', params=network_params)
    # cql_q1_current_actions, cql_q2_current_actions = agent.network(observations, cql_current_actions, goals, method='qf', params=network_params)
    # cql_q1_next_actions, cql_q2_next_actions = agent.network(observations, cql_next_actions, goals, method='qf', params=network_params)
    
    # # importance sample
    # random_density = jnp.log(0.5 ** agent.config['action_dim'])
    # cql_cat_q1 = jnp.concatenate([cql_q1_rand - random_density, cql_q1_next_actions - cql_next_log_pi , cql_q1_current_actions - cql_current_log_pi], axis=1)
    # cql_cat_q2 = jnp.concatenate([cql_q2_rand - random_density, cql_q2_next_actions - cql_next_log_pi, cql_q2_current_actions - cql_current_log_pi], axis=1)

    # cql_qf1_ood = (jax.scipy.special.logsumexp(cql_cat_q1 / agent.config['cql_temp'], axis=1)* agent.config['cql_temp'])
    # cql_qf2_ood = (jax.scipy.special.logsumexp(cql_cat_q2 / agent.config['cql_temp'], axis=1)* agent.config['cql_temp'])
    
    # cql_qf1_diff = jnp.clip(cql_qf1_ood - q1.mean(), agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
    # cql_qf2_diff = jnp.clip(cql_qf2_ood - q2.mean(), agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
    
    # # low alpha prime
    # log_low_alpha_prime = agent.network(method='log_alpha_prime')
    # low_alpha_prime = jnp.clip(jnp.exp(log_low_alpha_prime), 0.0, 1e6)
    # # low_alpha_prime = jax.lax.stop_gradient(low_alpha_prime)
    
    
    # cql_min_qf2_loss = low_alpha_prime * (cql_qf2_diff - agent.config['cql_low_target_action_gap'])
    # cql_min_qf1_loss = low_alpha_prime * (cql_qf1_diff - agent.config['cql_low_target_action_gap'])
                    
    # qf1_loss = q_loss1 + cql_min_qf1_loss
    # qf2_loss = q_loss2 + cql_min_qf2_loss
    
    # low_alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
    
    # q_loss = qf1_loss + qf2_loss + low_alpha_prime_loss
    # # q_loss = qf1_loss + qf2_loss 
    
    # # for alpha prime update
    # batch['cql_qf2_diff'] = cql_qf2_diff - agent.config['cql_low_target_action_gap']
    # batch['cql_qf1_diff'] = cql_qf1_diff - agent.config['cql_low_target_action_gap']
    
    return q_loss, {
        'q_loss': q_loss,
        'q max': q1.max(),
        'q min': q1.min(),
        'q mean': q1.mean(),
        'q bellman loss': q_loss1.mean(),
        # 'q cql_qf_diff': cql_qf1_diff.mean(),
        # 'q cql_min_qf_loss': cql_min_qf1_loss.mean(),
        # 'q cql_qf_ood': cql_qf1_ood.mean(),
        # 'q cql_q_rand': cql_q1_rand.mean(),
        # 'q cql_q_current_actions': cql_q1_current_actions.mean(),
        # 'q cql_q_next_actions': cql_q1_next_actions.mean(),
        # 'low_alpha_prime': low_alpha_prime,
        # 'low_alpha_prime_loss' : low_alpha_prime_loss
        

        
    }

def compute_high_qf_loss(agent, batch, network_params):
    # masks are 0 if terminal, 1 otherwise
    batch['high_masks'] = 1.0 - batch['high_rewards']
    # rewards are 0 if terminal, -1 otherwise
    batch['high_rewards'] = batch['high_rewards'] - 1.0
    
    if agent.config['high_action_in_hilp']:
        batch_size, obs_dim = batch['rep_observations'].shape
    else:   
        batch_size, obs_dim = batch['observations'].shape
    

    # sac-q loss
    # max target back up
    next_dist = agent.network(batch['high_targets'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=False, method='high_actor')
    next_actions, next_log_pi = supply_rng(next_dist.sample_and_log_prob)(sample_shape=10)
    
    (next_q1, next_q2) = agent.network(jnp.tile(batch['high_targets'], (10,1,1)), next_actions, jnp.tile(batch['high_goals'], (10,1,1)), method='high_target_qf')
    next_q_ = jnp.minimum(next_q1, next_q2)
    
    max_q_index = jnp.argmax(next_q_, axis=0)
    next_q_max = jnp.take_along_axis(next_q_.transpose(1,0), jnp.expand_dims(max_q_index, axis=-1), axis=-1) 
    next_log_pi_max = jnp.take_along_axis(next_log_pi.transpose(1,0), jnp.expand_dims(max_q_index, axis=-1), axis=-1) 
    
    # entropy 
    # log_alpha = agent.network(method='high_log_alpha')
    # alpha = jnp.exp(log_alpha)
    # next_q = next_q_max - alpha * next_log_pi_max
    
    # no entropy
    next_q = next_q_max
    
    target_q = batch['high_rewards'] + agent.config['discount'] * batch['high_masks'] * next_q
    target_q = jax.lax.stop_gradient(target_q)

    if agent.config['high_action_in_hilp']:
        # subgoals = (batch['rep_low_goals'] - batch['rep_observations']) / jnp.linalg.norm(batch['rep_low_goals'] - batch['rep_observations'], axis=-1, keepdims=True)
        rep_high_targets = batch['rep_high_targets']
        # rep_high_targets = batch['rep_high_targets'] / jnp.linalg.norm(batch['rep_high_targets'], axis=-1, keepdims=True)
        
        (q1, q2) = agent.network(batch['observations'], rep_high_targets, batch['high_goals'], method='high_qf', params=network_params)
    else:
        (q1, q2) = agent.network(batch['observations'], batch['high_targets'], batch['high_goals'], method='high_qf', params=network_params)

    q_loss1 = jnp.square(q1 - target_q).mean()
    q_loss2 = jnp.square(q2 - target_q).mean()
    

    # cql loss
    # dimension : (batch_size, cql_n_actions, obs_dim)
    observations =  extend_and_repeat(batch['observations'], 1, agent.config['cql_n_actions'])
    high_targets =  extend_and_repeat(batch['high_targets'], 1, agent.config['cql_n_actions'])
    high_goals =  extend_and_repeat(batch['high_goals'], 1, agent.config['cql_n_actions'])
    
    # random actions : (batch_size, cql_n_actions, action_dim)
    if agent.config['high_action_in_hilp']:
        # cql_random_actions = jax.random.uniform(key=agent.rng, shape=(batch_size, agent.config['cql_n_actions'], obs_dim),minval=1, maxval=1)
        cql_random_actions = jax.random.uniform(key=agent.rng, shape=(batch_size, agent.config['cql_n_actions'], obs_dim),minval=agent.config['rep_observation_min'], maxval=agent.config['rep_observation_max'])
    else:
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
    
    # original cql

    # key node  cql    
    if agent.config['key_node_q']:
        (key_q1, key_q2) = agent.network(batch['observations'], batch['high_target_key_node'], batch['high_goals'], method='high_qf', params=network_params)
        cql_qf1_diff = jnp.clip(cql_qf1_ood - (key_q1.mean()).mean(), agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
        cql_qf2_diff = jnp.clip(cql_qf2_ood - (key_q2.mean()).mean(), agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
    else:
        key_q1, key_q2 = jnp.zeros(0), jnp.zeros(0)
        cql_qf1_diff = jnp.clip(cql_qf1_ood - q1.mean(), agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
        cql_qf2_diff = jnp.clip(cql_qf2_ood - q2.mean(), agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
        
    
    # high alpha prime
    log_high_alpha_prime = agent.network(method='high_log_alpha_prime')
    high_alpha_prime = jnp.clip(jnp.exp(log_high_alpha_prime), 0.0, 1e6)
    
    # high_alpha_prime = jax.lax.stop_gradient(high_alpha_prime)
    
    cql_min_qf1_loss = high_alpha_prime * (cql_qf1_diff - agent.config['cql_high_target_action_gap'])
    cql_min_qf2_loss = high_alpha_prime * (cql_qf2_diff - agent.config['cql_high_target_action_gap'])
                    
    qf1_loss = q_loss1 + cql_min_qf1_loss
    qf2_loss = q_loss2 + cql_min_qf2_loss
    
    high_alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
    
    q_loss = qf1_loss + qf2_loss + high_alpha_prime_loss
    
    # for alpha prime update
    batch['cql_qf1_diff'] = cql_qf1_diff - agent.config['cql_high_target_action_gap']
    batch['cql_qf2_diff'] = cql_qf2_diff - agent.config['cql_high_target_action_gap']
    
    
    # q_loss = q_loss1 + q_loss2 
    
    
    return q_loss, {
        'q_loss': q_loss,
        'q max': q1.max(),
        'q min': q1.min(),
        'q mean': q1.mean(),
        'q key_q': key_q1.mean(),
        'q bellman loss': q_loss1.mean(),
        'q cql_qf_diff': cql_qf1_diff.mean(),
        'q cql_min_qf_loss': cql_min_qf1_loss.mean(),
        'q cql_qf_ood': cql_qf1_ood.mean(),
        'q cql_q_rand': cql_q1_rand.mean(),
        'q cql_q_current_actions': cql_q1_current_actions.mean(),
        'q cql_q_next_actions': cql_q1_next_actions.mean(),
        'high_alpha_prime': high_alpha_prime,
        'high_alpha_prime_loss' : high_alpha_prime_loss
    }
    
def compute_low_alpha_loss(agent, batch, network_params):
    # low log alpha
    # dist = agent.network(batch['observations'], batch['goals'], state_rep_grad=True, goal_rep_grad=False, method='actor')
    # actions, low_log_pi = supply_rng(dist.sample_and_log_prob)()
    low_log_pi = jax.lax.stop_gradient(batch['low_log_pi'])
    
    log_alpha = agent.network(method='log_alpha', params=network_params)
    low_alpha_loss = -log_alpha * (low_log_pi + agent.config['target_entropy']).mean()
    # low_alpha_loss = -log_alpha * (low_log_pi).mean()
    
    return low_alpha_loss, {
        'low_alpha_loss': low_alpha_loss,
        'low_alpha' : jnp.exp(log_alpha),
        }

def compute_high_alpha_loss(agent, batch, network_params):
    # high log alpha
    # high_dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=False, method='high_actor')
    # actions, high_log_pi = supply_rng(high_dist.sample_and_log_prob)()
    # high_log_pi = jax.lax.stop_gradient(batch['high_log_pi'])
    # log_high_alpha = agent.network(method='high_log_alpha', params=network_params)
    # high_alpha_loss = -log_high_alpha * (high_log_pi + agent.config['high_target_entropy']).mean()
    
    
    # kl loss alpha
    high_alpha = jnp.clip(jnp.exp(agent.network(method='high_log_alpha', params=network_params)), 0, 1e6)
    high_log_pi = jax.lax.stop_gradient(batch['high_kl_loss'])
    high_alpha_loss = high_alpha * (agent.config['high_target_divergence'] - high_log_pi).mean()
    
    
    return high_alpha_loss, {
        'high_alpha_loss': high_alpha_loss,
        # 'high_alpha' : jnp.exp(log_high_alpha),
        'high_alpha' : high_alpha,
        }
    
def compute_low_alpha_prime_loss(agent, batch, network_params):
    cql_qf1_diff = jax.lax.stop_gradient(batch['cql_qf1_diff'])
    cql_qf2_diff = jax.lax.stop_gradient(batch['cql_qf2_diff'])
    
    # high alpha prime
    log_alpha_prime = agent.network(method='log_alpha_prime', params=network_params)
    alpha = jnp.clip(jnp.exp(log_alpha_prime), 0, 1e6)
    cql_min_qf1_loss = alpha * cql_qf1_diff
    cql_min_qf2_loss = alpha * cql_qf2_diff
    
    low_alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
    
    return low_alpha_prime_loss , {
        'low_alpha_prime' : jnp.exp(log_alpha_prime)
    }

def compute_high_alpha_prime_loss(agent, batch, network_params):
    cql_qf1_diff = jax.lax.stop_gradient(batch['cql_qf1_diff'])
    cql_qf2_diff = jax.lax.stop_gradient(batch['cql_qf2_diff'])
    
    # high alpha prime
    log_high_alpha_prime = agent.network(method='high_log_alpha_prime', params=network_params)
    alpha = jnp.clip(jnp.exp(log_high_alpha_prime), 0, 1e6)
    cql_min_qf1_loss = alpha * cql_qf1_diff
    cql_min_qf2_loss = alpha * cql_qf2_diff
    
    high_alpha_prime_loss =  (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
    
    return high_alpha_prime_loss , {
        'high_alpha_prime' : jnp.exp(log_high_alpha_prime)
    }
    
def hilp_compute_value_loss(agent, batch, network_params):
    masks = 1.0 - batch['rewards']
    rewards = batch['rewards'] - 1.0
    
    (next_v1, next_v2) = agent.network(batch['next_observations'], batch['goals'], method='hilp_target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    (v1_t, v2_t) = agent.network(batch['observations'], batch['goals'], method='hilp_target_value')
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = rewards + agent.config['discount'] * masks * next_v1
    q2 = rewards + agent.config['discount'] * masks * next_v2
    (v1, v2) = agent.network(batch['observations'], batch['goals'], method='hilp_value', params=network_params)
    v = (v1 + v2) / 2

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['pretrain_expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['pretrain_expectile']).mean()
    
    if agent.config['n_step_hilp']:
        if agent.config['distance'] == 'log': # 'distance_log'
            # obs -> target : value
            (obs_target_v1, obs_target_v2) = agent.network(batch['observations'], batch['low_goals'], method='hilp_value', params=network_params)
            obs_target_v = jnp.minimum(obs_target_v1, obs_target_v2)
            distance_obs_target_v = jnp.log(jnp.maximum(1-obs_target_v, 1e-6))
            # target -> goals : value
            (target_goal_v1, target_goal_v2) = agent.network(batch['low_goals'], batch['goals'], method='hilp_target_value', params=network_params)
            target_goal_v = jnp.minimum(target_goal_v1, target_goal_v2)
            distance_target_goal_v = jnp.log(jnp.maximum(1-target_goal_v, 1e-6))
            # obs -> next_obs : value
            (obs_next_obs_v1, obs_next_obs_v2) = agent.network(batch['observations'], batch['next_observations'], method='hilp_target_value')
            obs_next_obs_v = jnp.minimum(obs_next_obs_v1, obs_next_obs_v2)
            distance_obs_next_obs_v = jnp.log(jnp.maximum(1-obs_next_obs_v, 1e-6))
            
            distance_next_v = jnp.log(jnp.maximum(1-next_v, 1e-6))
            
            n_step_loss = jnp.mean(((distance_obs_target_v + distance_target_goal_v) - (distance_obs_next_obs_v + distance_next_v))**2)
            
        elif agent.config['distance'] =='first':
            # obs -> target : value
            (obs_target_v1, obs_target_v2) = agent.network(batch['observations'], batch['low_goals'], method='hilp_value', params=network_params)
            obs_target_v = jnp.minimum(obs_target_v1, obs_target_v2)
            # obs -> next_obs : value
            # (obs_next_obs_v1, obs_next_obs_v2) = agent.network(batch['observations'], batch['next_observations'], method='hilp_target_value')
            # obs_next_obs_v = jnp.minimum(obs_next_obs_v1, obs_next_obs_v2)
            # next -> tareget : value
            (next_obs_target_v1, next_obs_target_v2) = agent.network(batch['next_observations'], batch['low_goals'], method='hilp_target_value')
            next_obs_target_v = jnp.minimum(next_obs_target_v1, next_obs_target_v2)
            
            #first
            n_step_loss = jnp.mean((obs_target_v  - (batch['rewards'] + next_obs_target_v))**2)
            
        elif agent.config['distance'] =='second':
            # obs -> target : value
            (obs_target_v1, obs_target_v2) = agent.network(batch['observations'], batch['low_goals'], method='hilp_value', params=network_params)
            obs_target_v = jnp.minimum(obs_target_v1, obs_target_v2)
            # target -> goals : value
            (target_goal_v1, target_goal_v2) = agent.network(batch['low_goals'], batch['goals'], method='hilp_target_value', params=network_params)
            target_goal_v = jnp.minimum(target_goal_v1, target_goal_v2)
            # obs -> next_obs : value
            # (obs_next_obs_v1, obs_next_obs_v2) = agent.network(batch['observations'], batch['next_observations'], method='hilp_target_value')
            # obs_next_obs_v = jnp.minimum(obs_next_obs_v1, obs_next_obs_v2)
            # next -> target : value
            # (next_obs_target_v1, next_obs_target_v2) = agent.network(batch['next_observations'], batch['low_goals'], method='hilp_target_value')
            # next_obs_target_v = jnp.minimum(next_obs_target_v1, next_obs_target_v2)
            
            # second
            # n_step_loss_1 = jnp.mean((obs_target_v  - (obs_next_obs_v + next_obs_target_v))**2)
            # n_step_loss_2 = jnp.mean(((obs_target_v + target_goal_v) - (obs_next_obs_v + next_v))**2)
            n_step_loss = jnp.mean(((obs_target_v + target_goal_v) - (batch['rewards'] + next_v))**2)
            # n_step_loss = n_step_loss_1 + n_step_loss_2
        else:
            raise NotImplementedError
    
        value_loss = value_loss1 + value_loss2 + n_step_loss
    else:
        value_loss = value_loss1 + value_loss2
        n_step_loss = 0
        
    return value_loss, {
        'value_loss': value_loss,
        'v max': v.max(),
        'v min': v.min(),
        'v mean': v.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'adv mean': adv.mean(),
        'adv max': adv.max(),
        'adv min': adv.min(),
        'accept prob': (adv >= 0).mean(),
        'n_step_loss': n_step_loss,
        'value_loss_only': value_loss1 + value_loss2,
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
            
            if hilp_update:
                qf_update, actor_update, high_actor_update = False, False, False
            else:
                qf_update, actor_update, high_actor_update = True, True, True
                
            # HILP Representation
            if hilp_update:
            # if agent.config['use_rep'] in ["hilp_subgoal_encoder", "hilp_encoder"]:
                hilp_value_loss, hilp_value_info = hilp_compute_value_loss(agent, pretrain_batch, network_params)
                for k, v in hilp_value_info.items():
                    info[f'hilp_value/{k}'] = v
            else:
                hilp_value_loss = 0.
            
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

            loss = qf_loss + high_qf_loss + actor_loss + high_actor_loss + hilp_value_loss

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
        new_params = unfreeze(agent.network.params)
        

        
        # HILP update
        # if agent.config['use_rep'] in ["hilp_subgoal_encoder", "hilp_encoder"]:
        if hilp_update:
            params = unfreeze(new_network.params)
            
            hilp_new_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_hilp_value'], agent.network.params['networks_hilp_target_value']
            )
            params['networks_hilp_target_value'] = hilp_new_target_params
            
            hilp_new_params = jax.tree_map(
            lambda op, np: np * agent.config['target_update_rate'] + op * (1 - agent.config['target_update_rate']), agent.network.params['networks_hilp_value'], new_network.params['networks_hilp_value']
            )
            params['networks_hilp_target_value'] = hilp_new_params
            new_network = new_network.replace(params=freeze(params))
        
        elif qf_update:
            # low alpha prime
            # network, low_alpha_prime_info = agent.network.apply_loss_fn(loss_fn=low_alpha_prime_loss_fn, has_aux=True)
            # info.update(low_alpha_prime_info)        
            # new_params['networks_log_alpha_prime'] = network.params['networks_log_alpha_prime']
            
            # high alpha prime
            network, high_alpha_prime_info = agent.network.apply_loss_fn(loss_fn=high_alpha_prime_loss_fn, has_aux=True)
            info.update(high_alpha_prime_info)
            new_params['networks_high_log_alpha_prime'] = network.params['networks_high_log_alpha_prime']
            
            # low alpha update
            # network, low_alpha_info = agent.network.apply_loss_fn(loss_fn=low_alpha_loss_fn, has_aux=True)
            # info.update(low_alpha_info)
            # new_params['networks_log_alpha'] = network.params['networks_log_alpha']
            
            # high alpha update
            network, high_alpha_info = agent.network.apply_loss_fn(loss_fn=high_alpha_loss_fn, has_aux=True)
            info.update(high_alpha_info)
            new_params['networks_high_log_alpha'] = network.params['networks_high_log_alpha']
            
            new_network = new_network.replace(params=freeze(new_params))
            # pass
        
        return agent.replace(network=new_network), info
        # return agent.replace(params=freeze(new_params)), info
    pretrain_update = jax.jit(pretrain_update, static_argnames=('qf_update', 'actor_update', 'alpha_update', 'high_actor_update', 'hilp_update'))

    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       subgoals: np.ndarray,
                       *,
                       low_dim_goals: bool = False,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = None,
                       visual: int = 0) -> jnp.ndarray:
        
        if visual:
            observations = agent.encoder(observations, method='encoder')
            if goals.shape[-1] != observations:
                goals = agent.encoder(goals, method='encoder')
                
        if subgoals is not None:
            dist = agent.network(observations, subgoals, goals, low_dim_goals=low_dim_goals, temperature=temperature, method='actor')
        else:
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
                            num_samples: int = None,
                            visual: int = 0) -> jnp.ndarray:
        if visual:
            observations = agent.encoder(observations, method='encoder')
            if goals.shape[-1] != observations:
                goals = agent.encoder(goals, method='encoder')
                
        dist = agent.network(observations, goals, temperature=temperature, method='high_actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        return actions
    sample_high_actions = jax.jit(sample_high_actions, static_argnames=('num_samples',))

    # HILP 
    @jax.jit
    def get_hilp_phi(agent,
                            *,
                            observations: jnp.ndarray) -> jnp.ndarray:
        return agent.network(observations=observations, method='hilp_phi')

    @jax.jit
    def get_hilp_value(agent,
                            *,
                            observations: jnp.ndarray,
                            goals: jnp.ndarray) -> jnp.ndarray:
        return agent.network(observations=observations, goals=goals, method='hilp_value')
    
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
        value_hidden_dims: Sequence[int] = (256, 256),
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
        cql_high_target_action_gap = 10,
        high_target_divergence = 1,        
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
        visual_encoder = None 
        
        if 'visual' in flag.env_name:
            # assert use_rep
            # from jaxrl_m.vision import encoders

            visual_encoder = get_encoder()
            # observations = visual_encoder(observations)
            
            # def make_encoder(bottleneck):
            #     if bottleneck:
            #         return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(rep_dim,), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=flag.rep_type, bottleneck=True)
            #     else:
            #         return RelativeRepresentation(rep_dim=qf_hidden_dims[-1], hidden_dims=(qf_hidden_dims[-1],), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=flag.rep_type, bottleneck=False)

            # qf_state_encoder = make_encoder(bottleneck=False)
            # qf_goal_encoder = make_encoder(bottleneck=use_waypoints)
            # high_qf_state_encoder = make_encoder(bottleneck=False)
            # high_qf_goal_encoder = make_encoder(bottleneck=use_waypoints)
            # policy_state_encoder = make_encoder(bottleneck=False)
            # policy_goal_encoder = make_encoder(bottleneck=False)
            # high_policy_state_encoder = make_encoder(bottleneck=False)
            # high_policy_goal_encoder = make_encoder(bottleneck=False)
        else:
            def make_encoder(bottleneck):
                if bottleneck:
                    return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(*qf_hidden_dims, rep_dim), layer_norm=use_layer_norm, rep_type=flag.rep_type, bottleneck=True)
                else:
                    return RelativeRepresentation(rep_dim=qf_hidden_dims[-1], hidden_dims=(*qf_hidden_dims, qf_hidden_dims[-1]), layer_norm=use_layer_norm, rep_type=flag.rep_type, bottleneck=False)

            if use_rep:
                qf_goal_encoder = make_encoder(bottleneck=True)

        qf_def = MonolithicQF(hidden_dims=qf_hidden_dims, use_layer_norm=use_layer_norm, bilinear=1)
        high_qf_def = MonolithicQF(hidden_dims=qf_hidden_dims, use_layer_norm=use_layer_norm, bilinear=1)
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

        high_action_dim = observations.shape[-1] if not flag.high_action_in_hilp else flag.hilp_skill_dim
        high_actor_def = Policy(actor_hidden_dims, action_dim=high_action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        hilp_value_goal_encoder = HILP_GoalConditionedPhiValue(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, ensemble=True, skill_dim=flag.hilp_skill_dim)

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
                'encoder' : visual_encoder,
            },
            networks={
                'hilp_value' : hilp_value_goal_encoder, # hilp
                'hilp_target_value' : copy.deepcopy(hilp_value_goal_encoder), # hilp
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
        
        
        config = flax.core.FrozenDict(**flag_dict,  **{'target_update_rate':tau, 'cql_n_actions':cql_n_actions, 'alpha_multiplier' : alpha_multiplier, 'use_automatic_entropy_tuning' : use_automatic_entropy_tuning, 'backup_entropy' : backup_entropy, 'policy_lr' : policy_lr, 'cql_min_q_weight' :cql_min_q_weight, 'qf_lr' : qf_lr, 'optimizer_type' : optimizer_type, 'soft_target_update_rate' : soft_target_update_rate, 'cql_n_actions' : cql_n_actions, 'cql_importance_sample' : cql_importance_sample, 'cql_lagrange' : cql_lagrange, 'cql_target_action_gap' : cql_target_action_gap, 'cql_temp' : cql_temp, 'cql_max_target_backup' : cql_max_target_backup, 'cql_clip_diff_min' : cql_clip_diff_min, 'cql_clip_diff_max' : cql_clip_diff_max, 'action_dim':action_dim, 'high_action_dim':high_action_dim, 'high_alpha_multiplier':high_alpha_multiplier, 'alpha_multiplier':alpha_multiplier, 'cql_low_target_action_gap':cql_low_target_action_gap, 'cql_high_target_action_gap': cql_high_target_action_gap, 'high_target_divergence': high_target_divergence})

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