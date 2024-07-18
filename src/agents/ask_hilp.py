import copy

from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, Critic, ensemblize, DiscretePolicy
from jaxrl_m.evaluation import supply_rng
from functools import partial


import flax
import flax.linen as nn
from flax.core import freeze, unfreeze
import ml_collections
from . import iql
from src.special_networks import Representation, HierarchicalActorCritic, RelativeRepresentation, MonolithicVF, Vae_Encoder, Vae_Decoder, HILP_GoalConditionedPhiValue


@jax.jit
def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

def compute_actor_loss(agent, batch, network_params):
    # if agent.config['use_rep'] == "vae_encoder":
    #     cur_goals, _, _ = agent.network(batch['low_goals'], method='vae_state_encoder', params=network_params)
    #     observations, _, _ = agent.network(batch['observations'], method='vae_state_encoder', params=network_params)
    #     next_observations, _, _ = agent.network(batch['next_observations'], method='vae_state_encoder', params=network_params)        
        
    # elif agent.config['use_rep'] == "hilp_encoder":
    #     cur_goals = agent.network(batch['low_goals'], method='hilp_phi')
    #     observations = agent.network(batch['observations'], method='hilp_phi')
    #     next_observations = agent.network(batch['next_observations'], method='hilp_phi')
        
    # elif agent.config['use_rep'] == "hilp_subgoal_encoder":
    #     cur_goals = agent.network(batch['low_goals'], method='hilp_phi')
    #     observations = batch['observations']
    #     next_observations = batch['next_observations'] 

    # else:
    # original low goal data
    cur_goals = batch['low_goals']
    observations = batch['observations']
    next_observations = batch['next_observations']
    final_goal = batch['high_goals']
        
    
    
    if agent.config['low_actor_train_with_high_actor']:
        high_dist = agent.network(observations, final_goal, temperature=agent.config['high_temperature'], method='high_actor')
        low_goals = supply_rng(high_dist.sample)()
    elif agent.config['high_action_in_hilp']:
        low_goals = batch['rep_low_goals']  
    else:
        low_goals = batch['low_goals'] 
        
    v1, v2 = agent.network(observations, cur_goals, method='hilp_value')
    nv1, nv2 = agent.network(next_observations, cur_goals, method='hilp_value')
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2

    adv = nv - v
    exp_a = jnp.exp(adv * agent.config['temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    if agent.config['final_goal']:
        # [obs, subgoal, final goal] input
        dist = agent.network(observations, low_goals, final_goal, state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
    else:
        # [obs, subgoal] input
        dist = agent.network(observations, low_goals, state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
        
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()

    # high actor debug
    high_action_xy_mse_loss, high_action_mse_loss = 0, 0

    if agent.config['low_actor_train_with_high_actor']:
        if agent.config['high_action_in_hilp']:
            high_targets = batch['rep_low_goals']  
        else:
            high_targets = batch['low_goals']  
        high_action_mse_loss = jnp.mean((high_dist.mode() - high_targets)**2)
        if 'ant' in agent.config['env_name'] and not agent.config['high_action_in_hilp']:
            high_action_xy_mse_loss = jnp.mean((high_dist.mode()[:,:2] - high_targets[:,:2])**2)

    
    return actor_loss, {
        'actor_loss': actor_loss,
        'adv': adv.mean(),
        'bc_log_probs': log_probs.mean(),
        'adv_median': jnp.median(adv),
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
        'high_action_mse_loss': high_action_mse_loss,
        'high_action_xy_mse_loss': high_action_xy_mse_loss,
    }

def compute_high_actor_loss(agent, batch, network_params):
    # if agent.config['use_rep'] == "vae_encoder":
    #     cur_goals, _, _ = agent.network(batch['high_goals'], method='vae_state_encoder', params=network_params)
    #     observations, _, _ = agent.network(batch['observations'], method='vae_state_encoder', params=network_params)
    #     #if agent.config['keynode_ratio']:
    #     #    length = int(len(observations)*agent.config['keynode_ratio'])
    #     #    batch['high_targets'] = jnp.concatenate([batch['high_targets'][:length], batch['key_node'][length:]], axis=0)
    #     high_targets, _, _ = agent.network(batch['high_targets'], method='vae_state_encoder', params=network_params)
        
    # elif agent.config['use_rep'] == "hilp_encoder":
    #     cur_goals = agent.network(batch['high_goals'], method='hilp_phi')
    #     observations = agent.network(batch['observations'], method='hilp_phi')
    #     high_targets = agent.network(batch['high_targets'] , method='hilp_phi')
    # elif agent.config['use_rep'] == "hilp_subgoal_encoder":
    #     cur_goals = agent.network(batch['high_goals'], method='hilp_phi')
    #     observations = batch['observations']
    #     high_targets = batch['high_targets']
    # else:
    cur_goals = batch['high_goals']
    observations = batch['observations']
    high_targets = batch['high_targets']
    
    v1, v2 = agent.network(observations, cur_goals, method='hilp_value')
    nv1, nv2 = agent.network(high_targets, cur_goals, method='hilp_value')
    
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2
    
    if agent.config['correction_value']:
        #devide
        diff_v1, diff_v2 = agent.network(observations, high_targets, method='hilp_value')
        diff_v_ = jnp.minimum(diff_v1, diff_v2)
        adv = (nv - v + diff_v_)/2
        # # min
        # diff_v1, diff_v2 = agent.network(observations, high_targets, method='value')
        # adv_ = nv - v
        # diff_v_ = jnp.minimum(diff_v1, diff_v2)
        # diff_v = jnp.maximum(diff_v_, 1e-6)
        # adv = jnp.abs(adv_ / diff_v) * adv_
    else:
        adv = nv - v
        

    exp_a = jnp.exp(adv * agent.config['high_temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    dist = agent.network(observations, cur_goals, state_rep_grad=True, goal_rep_grad=True, method='high_actor', params=network_params)

    if agent.config['high_action_in_hilp']:
        # latent obs train
        target = agent.network(high_targets, method='hilp_phi')
    else:
        # raw obs train
        target = high_targets - observations
    
    log_probs = dist.log_prob(target)
    actor_loss_ = -(exp_a * log_probs).mean()
    
    if agent.config['mse_loss']:
        high_actions = supply_rng(dist.sample)(sample_shape=batch['observations'].shape[0])
        if agent.config['high_action_in_hilp']:
            mse_loss = agent.config['mse_loss'] * jnp.mean((high_actions - (batch['key_node']))**2)
        else:
            mse_loss = agent.config['mse_loss'] * jnp.mean((high_actions - (batch['key_node'] - observations))**2)
    
        actor_loss_only = -(exp_a * (log_probs)).mean()
        actor_loss_ = -(exp_a * (log_probs - mse_loss)).mean()
        
    else:
        actor_loss_only = 0
        actor_loss_ = -(exp_a * log_probs).mean()
        mse_loss = 0
        
    
    
    kl_loss = 0
    if agent.config['kl_loss']:
        key_node_target = batch['key_node'] - observations
        key_node_log_probs = dist.log_prob(key_node_target)
        policy_log_probs = dist.mode()
            
        kl_loss = - (jnp.exp(key_node_log_probs) * (-key_node_log_probs - policy_log_probs)).mean()

        log_std_q = dist.log_std()
        log_std_p = key_node_target
        
        std_q = jnp.exp(log_std_q)
        std_p = jnp.exp(log_std_p)

        kl_div = log_std_p - log_std_q + (std_q ** 2 + (means_q - means_p) ** 2) / (2.0 * std_p ** 2) - 0.5

        
        
    actor_loss = actor_loss_ + kl_loss
        
    return actor_loss, {
        'high_actor_loss': actor_loss,
        'high_adv': adv.mean(),
        'high_bc_log_probs': log_probs.mean(),
        'high_adv_median': jnp.median(adv),
        'high_mse': jnp.mean((dist.mode() - target)**2),
        'high_scale': dist.scale_diag.mean(),
        'kl_loss' : kl_loss,
        'actor_loss_only' : actor_loss_only,
        'mse_loss' : mse_loss,
    }

def compute_value_loss(agent, batch, network_params):
    masks = 1.0 - batch['rewards']
    rewards = batch['rewards'] - 1.0
    
    if agent.config['use_rep'] == "vae_encoder":
        observations, _, _ = agent.network(batch['observations'], method='vae_state_encoder', params=network_params)
        next_observations, _, _ = agent.network(batch['next_observations'], method='vae_state_encoder', params=network_params)
        # 질문 왜 vae만 하는지여부
        #if agent.config['keynode_ratio']:
        #    length = int(agent.config['keynode_ratio']*len(observations))
        #    batch['goals'] = jnp.concatenate([batch['goals'][:length], batch['key_node'][length:]], axis=0)
        cur_goals, _, _ = agent.network(batch['goals'], method='vae_state_encoder', params=network_params)
    
    elif agent.config['use_rep'] == "hilp_encoder":
        cur_goals = agent.network(batch['goals'], method='hilp_phi')
        observations = agent.network(batch['observations'], method='hilp_phi')
        next_observations = agent.network(batch['next_observations'], method='hilp_phi')
        
    elif agent.config['use_rep'] == "hilp_subgoal_encoder":
        cur_goals = agent.network(batch['goals'], method='hilp_phi')
        observations = batch['observations']
        next_observations = batch['next_observations']

    else:
        cur_goals = batch['goals']
        observations = batch['observations']
        next_observations = batch['next_observations']
    
    (next_v1, next_v2) = agent.network(next_observations, cur_goals, method='target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = rewards + agent.config['discount'] * masks * next_v

    (v1_t, v2_t) = agent.network(observations, cur_goals, method='target_value')
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = rewards + agent.config['discount'] * masks * next_v1
    q2 = rewards + agent.config['discount'] * masks * next_v2
    (v1, v2) = agent.network(observations, cur_goals, method='value', params=network_params)

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['pretrain_expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['pretrain_expectile']).mean()
    value_loss = value_loss1 + value_loss2

    advantage = adv
    return value_loss, {
        'value_loss': value_loss,
        'v max': v1.max(),
        'v min': v1.min(),
        'v mean': v1.mean(),
        'abs adv mean': jnp.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
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
        # obs -> target : value
        (obs_target_v1, obs_target_v2) = agent.network(batch['observations'], batch['low_goals'], method='hilp_value', params=network_params)
        obs_target_v = jnp.minimum(obs_target_v1, obs_target_v2)
        
        # target -> goals : value
        (target_goal_v1, target_goal_v2) = agent.network(batch['low_goals'], batch['goals'], method='hilp_target_value')
        target_goal_v = jnp.minimum(target_goal_v1, target_goal_v2)
        
        # target -> goals : value
        (obs_next_obs_v1, obs_next_obs_v2) = agent.network(batch['observations'], batch['next_observations'], method='hilp_target_value')
        obs_next_obs_v = jnp.minimum(obs_next_obs_v1, obs_next_obs_v2)
        
        
        n_step_loss = jnp.mean(((obs_target_v + target_goal_v) - (obs_next_obs_v + next_v))**2)
    
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
    }
    
def vae_recon_loss(agent, batch, network_params):
    """
    ELBO: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)
    KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
            = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
            = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
    """
    observations = batch['observations']
    z, mean, std = agent.network(observations, method = 'vae_state_encoder', params=network_params)
    output = agent.network(z, method = 'vae_state_decoder', params=network_params)
    var = jnp.square(std)
    rc_loss = -jnp.mean(jnp.square(observations - output), axis=-1)
    kl = 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)
    vae_loss = rc_loss - agent.config['vae_kl_coe'] * kl
    vae_loss = -jnp.mean(vae_loss)
    return vae_loss, {
        'vae_loss': vae_loss,
        'reconstruction_loss': rc_loss.mean(),
        'kl': kl.mean(),
        'std' : std.mean(),
        'log_var' : jnp.log(var).mean(),
        'mean' : mean.mean()
    }
    
class JointTrainAgent(iql.IQLAgent):
    network: TrainState = None
    key_nodes : dict = None
    
    def pretrain_update(agent, pretrain_batch, seed=None, value_update=True, actor_update=True, high_actor_update=True, hilp_update=True):
        def loss_fn(network_params):
            info = {}
            if hilp_update:
                value_update, actor_update, high_actor_update = False, False, False
            else:
                value_update, actor_update, high_actor_update = True, True, True
                
            # Value
            if value_update:
                value_loss, value_info = compute_value_loss(agent, pretrain_batch, network_params)
                for k, v in value_info.items():
                    info[f'value/{k}'] = v
            else:
                value_loss = 0.

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
            
            # HILP Representation
            if hilp_update:
            # if agent.config['use_rep'] in ["hilp_subgoal_encoder", "hilp_encoder"]:
                hilp_value_loss, hilp_value_info = hilp_compute_value_loss(agent, pretrain_batch, network_params)
                for k, v in hilp_value_info.items():
                    info[f'hilp_value/{k}'] = v
            else:
                hilp_value_loss = 0.
                
            # VAE-Reconstrunction
            if agent.config['use_rep'] == "vae_encoder":
                rc_loss, rc_info = vae_recon_loss(agent, pretrain_batch, network_params)
                for k, v in rc_info.items():
                    info[f'recon/{k}'] = v
            else:
                rc_loss = 0.
                            
            loss = value_loss + actor_loss + high_actor_loss + hilp_value_loss + agent.config['vae_recon_coe']*rc_loss 
            
            return loss, info
        
        # HIQL/HILP/VAE update (기울기 구한 것만 (params=network_params))
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        
        # HIQL update
        if value_update:
            new_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
            )
            params = unfreeze(new_network.params)
            params['networks_target_value'] = new_target_params
            new_network = new_network.replace(params=freeze(params))
 
        # HILP update
        # if agent.config['use_rep'] in ["hilp_subgoal_encoder", "hilp_encoder"]:
        if hilp_update:
            hilp_new_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_hilp_value'], agent.network.params['networks_hilp_target_value']
            )
            params = unfreeze(new_network.params)
            params['networks_hilp_target_value'] = hilp_new_target_params
            new_network = new_network.replace(params=freeze(params))
            
        # VAE update (질문: 업데이트 방식 바꿔야할듯? 아래 업데이트 하는 이유는 무엇인가?)
        if agent.config['use_rep'] == "vae_encoder":
            new_encoder_params = jax.tree_map(lambda o,n : n * agent.config['target_update_rate'] + o * (1 - agent.config['target_update_rate']), agent.network.params['encoders_state_encoder'], new_network.params['encoders_state_encoder'])
            new_decoder_params = jax.tree_map(lambda o,n : n * agent.config['target_update_rate'] + o * (1 - agent.config['target_update_rate']), agent.network.params['encoders_state_decoder'], new_network.params['encoders_state_decoder'])
            params = unfreeze(new_network.params)
            params['encoders_state_encoder'] = new_encoder_params
            params['encoders_state_decoder'] = new_decoder_params
            new_network = new_network.replace(params=freeze(params))
            
        return agent.replace(network=new_network), info
    pretrain_update = jax.jit(pretrain_update, static_argnames=('value_update', 'actor_update', 'high_actor_update', 'hilp_update'))

    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       *,
                       low_dim_goals: bool = False,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       num_samples: int = None) -> jnp.ndarray:
        dist = agent.network(observations, goals, low_dim_goals=low_dim_goals, temperature=temperature, method='actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        actions = jnp.clip(actions, -1, 1)
        return actions
    sample_actions = jax.jit(sample_actions, static_argnames=('num_samples', 'low_dim_goals'))

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
        return agent.network(targets=targets, goals=bases, method='policy_goal_encoder')

    # HILP 
    # @jax.jit
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
    def get_value_goal(agent,
                            *,
                            targets: np.ndarray,
                            bases: np.ndarray = None,
                            ) -> jnp.ndarray:
        return agent.network(targets=targets, bases=bases, method='value_goal_encoder')
    
    # VAE    
    @jax.jit
    def get_vae_state_rep(agent,
                            *,
                            observation: jnp.ndarray) -> jnp.ndarray:
        return agent.network(targets=observation, method='vae_state_encoder')
    @jax.jit
    def get_vae_rep_state(agent,
                            *,
                            latent: np.ndarray,
                            ) -> jnp.ndarray:
        return agent.network(targets=latent, method='vae_state_decoder')
        
def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        state_hidden_dims: Sequence[int] = (128, 64),
        discount: float = 0.99,
        tau: float = 0.005,
        temperature: float = 1,
        high_temperature: float = 1,
        pretrain_expectile: float = 0.7,
        way_steps: int = 0,
        rep_dim: int = 10,
        use_layer_norm: int = 1,
        key_nodes : Any = None,
        flag : Any =None,
        **kwargs):
        print('Extra kwargs:', kwargs)
        
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, high_actor_key, critic_key, value_key = jax.random.split(rng, 5)

        value_goal_encoder = None # numerical or img
        hilp_value_goal_encoder = None # HILP-numerical 
        vae_state_encoder = None # vae-numerical 
        vae_state_decoder = None # vae-numerical 
        value_state_encoder = None # img
        policy_state_encoder = None # img
        policy_goal_encoder = None # img
        high_policy_state_encoder = None # img
        high_policy_goal_encoder = None # img
        
        encoder = None
                
        value_def = MonolithicVF(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm)
        action_dim = actions.shape[-1]
        actor_def = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)
        
        def make_encoder(bottleneck):
            # 0610 승호수정 goal only
            if bottleneck:
                return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(*value_hidden_dims, rep_dim), layer_norm=use_layer_norm, bottleneck=flag.rep_normalizing_On, rep_type=flag.rep_type)
            else:
                return RelativeRepresentation(rep_dim=value_hidden_dims[-1], hidden_dims=(*value_hidden_dims, value_hidden_dims[-1]), layer_norm=use_layer_norm, bottleneck=False, rep_type=flag.rep_type)
        
        # if flag.use_rep == 'hiql_goal_encoder':
        #     value_goal_encoder = make_encoder(bottleneck=True)
        
        # elif flag.use_rep in ['hilp_subgoal_encoder', 'hilp_encoder']: 
        #     hilp_value_goal_encoder = HILP_GoalConditionedPhiValue(hidden_dims=(512, 512, 512), use_layer_norm=use_layer_norm, ensemble=True, skill_dim=flag.hilp_skill_dim, encoder=False)
            
        # elif flag.use_rep == "vae_encoder":
        #     value_goal_encoder = make_encoder(bottleneck=True)
        #     vae_state_encoder = Vae_Encoder(rep_dim=rep_dim, hidden_dim=state_hidden_dims, layer_norm=use_layer_norm)
        #     vae_state_decoder = Vae_Decoder(hidden_dim=state_hidden_dims[::-1], layer_norm=use_layer_norm, output_shape=observations.shape[-1])
        
        if flag.high_action_in_hilp :
            high_action_dim = flag.hilp_skill_dim
        else:
            high_action_dim = observations.shape[-1]
            
        subgoals = jnp.zeros((1,high_action_dim))
        
        hilp_value_goal_encoder = HILP_GoalConditionedPhiValue(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, ensemble=True, skill_dim=flag.hilp_skill_dim, encoder=encoder)
        # hilp_value_goal_encoder = HILP_GoalConditionedPhiValue(hidden_dims=(512, 512, 512), use_layer_norm=use_layer_norm, ensemble=True, skill_dim=flag.hilp_skill_dim, encoder=False)
        
        
        high_actor_def = Policy(actor_hidden_dims, action_dim=high_action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        network_def = HierarchicalActorCritic(
            encoders={                
                'vae_state_encoder' : vae_state_encoder, # vae
                'vae_state_decoder' : vae_state_decoder, # vae
                'value_state': value_state_encoder,
                'value_goal': value_goal_encoder,
                'policy_state': policy_state_encoder,
                'policy_goal': policy_goal_encoder,
                'high_policy_state': high_policy_state_encoder,
                'high_policy_goal': high_policy_goal_encoder,
            },
            networks={
                'hilp_value' : hilp_value_goal_encoder, # hilp
                'hilp_target_value' : copy.deepcopy(hilp_value_goal_encoder), # hilp
                'value': value_def,
                'target_value': copy.deepcopy(value_def),
                'actor': actor_def,
                'high_actor': high_actor_def,
            },
            flag=flag,
        )
        network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
        network_params = network_def.init(value_key, observations=observations, goals=observations, subgoals=subgoals)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = unfreeze(network.params)
        params['networks_target_value'] = params['networks_value']
        
        if flag.use_rep in ["hilp_subgoal_encoder", "hilp_encoder"]: # hilp
            params['networks_hilp_target_value'] = params['networks_hilp_value'] # hilp

        network = network.replace(params=freeze(params))
        # config = flax.core.FrozenDict(dict(
        #     discount=discount, temperature=temperature, high_temperature=high_temperature,
        #     target_update_rate=tau, pretrain_expectile=pretrain_expectile, way_steps=way_steps, keynode_ratio=flag.keynode_ratio, 
        #     env_name=kwargs['env_name'], use_rep=flag.use_rep, vae_recon_coe=flag.vae_recon_coe, vae_kl_coe=flag.vae_kl_coe, kl_loss=flag.kl_loss
        # ))
        
        flag_dict = flag.flag_values_dict()
        config = flax.core.FrozenDict(**flag_dict, **{'target_update_rate':tau})

        return JointTrainAgent(rng, network=network, critic=None, value=None, target_value=None, actor=None, config=config, key_nodes=key_nodes)

def get_default_config():
    config = ml_collections.ConfigDict({
        'lr': 3e-4,
        'actor_hidden_dims': (256, 256),
        'value_hidden_dims': (256, 256),
        'discount': 0.99,
        'temperature': 1.0,
        'tau': 0.005,
        'pretrain_expectile': 0.7,
    })
    return config