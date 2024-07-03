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

# def alpha_loss(agent, ):
    
#     dist = agent.network(batch['observations'], batch['low_goals'], state_rep_grad=True, goal_rep_grad=True, method='actor')
#     samples, log_pi = supply_rng(action_distribution.sample_and_log_prob)()
    
#     if self.config.use_automatic_entropy_tuning:
#         alpha_loss = -self.log_alpha.apply(train_params['log_alpha']) * (log_pi + self.config.target_entropy).mean()
#         loss_collection['log_alpha'] = alpha_loss
#         alpha = jnp.exp(self.log_alpha.apply(train_params['log_alpha'])) * self.config.alpha_multiplier
#     else:
#         alpha_loss = 0.0
#         alpha = self.config.alpha_multiplier
#     return alpha_loss

# def cql_lagrange(agent, ):
#     pass 
#     return lagrange_loss


def compute_actor_loss(agent, batch, network_params):
    # if agent.config['use_waypoints']:  # Use waypoint states as goals (for hierarchical policies)
    #     cur_goals = batch['low_goals']
    # else:  # Use randomized last observations as goals (for flat policies)
    #     cur_goals = batch['high_goals']
    dist = agent.network(batch['observations'], batch['low_goals'], state_rep_grad=True, goal_rep_grad=False, method='actor', params=network_params)
    new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
    
    q1, q2 = agent.network(batch['observations'], new_actions, batch['low_goals'], method='qf')
    q = jnp.min(q1, q2)
    
    # nq1, nq2 = agent.network(batch['next_observations'], new_actions, batch['low_goals'], method='qf')
    # q = (q1 + q2) / 2
    # nq = (nq1 + nq2) / 2

    # adv = nq - q
    # exp_a = jnp.exp(adv * agent.config['temperature'])
    # exp_a = jnp.minimum(exp_a, 100.0)

    # if agent.config['use_waypoints']:
    #     goal_rep_grad = agent.config['policy_train_rep']
    # else:
    #     goal_rep_grad = True
    # dist = agent.network(batch['observations'], cur_goals, state_rep_grad=True, goal_rep_grad=goal_rep_grad, method='actor', params=network_params)
    # log_probs = dist.log_prob(batch['actions'])
    actor_loss = (alpha * log_pi - q).mean()

    return actor_loss, {
        'actor_loss': actor_loss,
        # 'adv': adv.mean(),
        # 'bc_log_probs': log_probs.mean(),
        # 'adv_median': jnp.median(adv),
        'q' : q.mean(),
        'log_pi' : log_pi.mean(),
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
    }


def compute_high_actor_loss(agent, batch, network_params):
    dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=True, method='high_actor', params=network_params)
    new_actions, log_pi = supply_rng(dist.sample_and_log_prob)()
    
    q1, q2 = agent.network(batch['observations'], new_actions, batch['high_goals'], method='high_qf')
    q = jnp.min(q1, q2)
    # nq1, nq2 = agent.network(batch['high_targets'], batch['high_goals'], method='high_qf')
    # v = (v1 + v2) / 2
    # nv = (nv1 + nv2) / 2

    # adv = nv - v
    # exp_a = jnp.exp(adv * agent.config['high_temperature'])
    # exp_a = jnp.minimum(exp_a, 100.0)
    if agent.config['alpha_multiplier']:
        alpha = agent.config['alpha_multiplier']
    else:
        alpha = jnp.exp(self.log_alpha.apply(train_params['log_alpha'])) * self.config.alpha_multiplier

    # dist = agent.network(batch['observations'], batch['high_goals'], state_rep_grad=True, goal_rep_grad=True, method='high_actor', params=network_params)
    # if agent.config['use_rep']:
    #     target = agent.network(targets=batch['high_targets'], bases=batch['observations'], method='qf_goal_encoder')
    # else:
    #     target = batch['high_targets'] - batch['observations']
    # log_probs = dist.log_prob(target)
    # actor_loss = -(exp_a * log_probs).mean()

    return actor_loss, {
        'high_actor_loss': actor_loss,
        # 'high_adv': adv.mean(),
        # 'high_bc_log_probs': log_probs.mean(),
        # 'high_adv_median': jnp.median(adv),
        'q' : q.mean(),
        'log_pi' : log_pi.mean(),
        'high_mse': jnp.mean((dist.mode() - target)**2),
        'high_scale': dist.scale_diag.mean(),
    }


def compute_qf_loss(agent, batch, network_params):
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = 1.0 - batch['rewards']
    # rewards are 0 if terminal, -1 otherwise
    batch['rewards'] = batch['rewards'] - 1.0

    # sac-q loss
    dist = agent.network(batch['next_observations'], batch['goals'], state_rep_grad=True, goal_rep_grad=False, method='actor')
    next_actions, next_log_pi = supply_rng(dist.sample_and_log_prob)()
    
    (next_q1, next_q2) = agent.network(batch['next_observations'], next_actions, batch['goals'], method='target_qf') 
    
    alpha = jnp.exp(agent.network(method='log_alpha')) * agent.config['alpha_multiplier']
    next_q = jnp.minimum(next_q1, next_q2) + alpha * next_log_pi
    
    target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
    target_q = jax.lax.stop_gradient(target_q)

    (q1, q2) = agent.network(batch['observations'], batch['actions'], batch['goals'], method='qf', params=network_params)

    q_loss1 = (jnp.square(q1 - target_q)).mean()
    q_loss2 = (jnp.square(q2 - target_q)).mean()


    # cql loss
    batch_size = batch['observations'].shape[0]
    # dimension : (batch_size, cql_n_actions, obs_dim)
    observations =  extend_and_repeat(batch['observations'], 1, agent.config['cql_n_actions'])
    next_observations =  extend_and_repeat(batch['next_observations'], 1, agent.config['cql_n_actions'])
    goals =  extend_and_repeat(batch['goals'], 1, agent.config['cql_n_actions'])
    
    # random actions : (batch_size, cql_n_actions, action_dim)
    cql_random_actions = jax.random.uniform(key=jax.random.PRNGKey(0), shape=(batch_size, agent.config['cql_n_actions'], agent.config['action_dim']),minval=agent.config['action_min'], maxval=agent.config['action_max'])
    dist = agent.network(observations, goals, state_rep_grad=True, goal_rep_grad=False, method='actor')
    cql_current_actions, cql_current_log_pi = supply_rng(dist.sample_and_log_prob)()
    
    dist = agent.network(next_observations, goals, state_rep_grad=True, goal_rep_grad=False, method='actor')
    cql_next_actions, cql_next_log_pi = supply_rng(dist.sample_and_log_prob)()
    
    
    cql_q1_rand, cql_q2_rand = agent.network(observations, cql_random_actions, goals, method='qf', params=network_params)
    cql_q1_current_actions, cql_q2_current_actions = agent.network(observations, cql_current_actions, goals, method='qf', params=network_params)
    cql_q1_next_actions, cql_q2_next_actions = agent.network(observations, cql_next_actions, goals, method='qf', params=network_params)
    
    cql_cat_q1 = jnp.concatenate([cql_q1_rand, jnp.expand_dims(q1, 1), cql_q1_next_actions, cql_q1_current_actions], axis=1)
    cql_cat_q2 = jnp.concatenate([cql_q2_rand, jnp.expand_dims(q2, 1), cql_q2_next_actions, cql_q2_current_actions], axis=1)

    cql_qf1_ood = (jax.scipy.special.logsumexp(cql_cat_q1 / agent.config['cql_temp'], axis=1)* agent.config['cql_temp'])
    cql_qf2_ood = (jax.scipy.special.logsumexp(cql_cat_q2 / agent.config['cql_temp'], axis=1)* agent.config['cql_temp'])
    
    cql_qf1_diff = jnp.clip(cql_qf1_ood - q1, agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
    cql_qf2_diff = jnp.clip(cql_qf2_ood - q2, agent.config['cql_clip_diff_min'], agent.config['cql_clip_diff_max'],).mean()
                
    cql_min_qf1_loss = cql_qf1_diff * agent.config['cql_min_q_weight']
    cql_min_qf2_loss = cql_qf2_diff * agent.config['cql_min_q_weight']
                    
    qf1_loss = q_loss1 + cql_min_qf1_loss
    qf2_loss = q_loss2 + cql_min_qf2_loss
    
    q_loss = qf1_loss + qf2_loss
    return q_loss, {
        'q_loss': q_loss,
        'q max': q1.max(),
        'q min': q1.min(),
        'q mean': q1.mean(),
        # 'abs adv mean': jnp.abs(advantage).mean(),
        # 'adv mean': advantage.mean(),
        # 'adv max': advantage.max(),
        # 'adv min': advantage.min(),
        # 'accept prob': (advantage >= 0).mean(),
    }
    
    
def compute_alpha_loss(agent, batch, network_params):
    dist = agent.network(batch['observations'], batch['goals'], state_rep_grad=True, goal_rep_grad=False, method='actor')
    actions, log_pi = supply_rng(dist.sample_and_log_prob)()
    
    log_alpha = agent.network(method='log_alpha')
    alpha_loss = -log_alpha * (log_pi + agent.config['target_entropy']).mean()
    alpha = jnp.exp(agent.network(method='log_alpha')) * agent.config['alpha_multiplier']

    return alpha_loss, {
        'alpha_loss': alpha_loss,
        'log_alpha' : log_alpha,
        'alpha' : alpha,
        }

class JointTrainAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    qf: TrainState
    target_qf: TrainState
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)
    network: TrainState = None
    key_nodes: dict = None
    
    def pretrain_update(agent, pretrain_batch, seed=None, qf_update=True, actor_update=True, high_actor_update=True):
        def loss_fn(network_params):
            info = {}

            
            
            # Q function
            if qf_update:
                qf_loss, qf_info = compute_qf_loss(agent, pretrain_batch, network_params)
                for k, v in qf_info.items():
                    info[f'qf/{k}'] = v
            else:
                qf_loss = 0.

            # Actor
            if actor_update:
                actor_loss, actor_info = compute_actor_loss(agent, pretrain_batch, network_params)
                for k, v in actor_info.items():
                    info[f'actor/{k}'] = v
            else:
                actor_loss = 0.

            # High Actor
            # if high_actor_update and agent.config['use_waypoints']:
            #     high_actor_loss, high_actor_info = compute_high_actor_loss(agent, pretrain_batch, network_params)
            #     for k, v in high_actor_info.items():
            #         info[f'high_actor/{k}'] = v
            # else:
            #     high_actor_loss = 0.

            loss = qf_loss + actor_loss
            # loss = qf_loss + actor_loss + high_actor_loss

            return loss, info

        if qf_update:
            new_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_qf'], agent.network.params['networks_target_qf']
            )

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        if qf_update:
            params = unfreeze(new_network.params)
            params['networks_target_qf'] = new_target_params
            new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info
    # pretrain_update = jax.jit(pretrain_update, static_argnames=('qf_update', 'actor_update', 'high_actor_update'))

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
        goals: jnp.ndarray,
        actions: jnp.ndarray,
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
        alpha_multiplier = 1.0,
        use_automatic_entropy_tuning = True,
        backup_entropy = False,
        target_entropy = 0.0,
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
        **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, high_actor_key, critic_key, qf_key = jax.random.split(rng, 5)

        qf_state_encoder = None
        qf_goal_encoder = None
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
        log_alpha = Scalar(0.0)
        
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
                'policy_state': policy_state_encoder,
                'policy_goal': policy_goal_encoder,
                'high_policy_state': high_policy_state_encoder,
                'high_policy_goal': high_policy_goal_encoder,
            },
            networks={
                'qf': qf_def,
                'target_qf': copy.deepcopy(qf_def),
                'actor': actor_def,
                'high_actor': high_actor_def,
                'log_alpha' : log_alpha,
            },
            flag=flag,
        )
        network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
        network_params = network_def.init(qf_key, observations, actions, goals)['params']
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
        config = flax.core.FrozenDict(**flag_dict, **{'target_update_rate':tau, 'cql_n_actions':cql_n_actions, 'alpha_multiplier' : alpha_multiplier, 'use_automatic_entropy_tuning' : use_automatic_entropy_tuning, 'backup_entropy' : backup_entropy, 'target_entropy' : target_entropy, 'policy_lr' : policy_lr, 'cql_min_q_weight' :cql_min_q_weight, 'qf_lr' : qf_lr, 'optimizer_type' : optimizer_type, 'soft_target_update_rate' : soft_target_update_rate, 'cql_n_actions' : cql_n_actions, 'cql_importance_sample' : cql_importance_sample, 'cql_lagrange' : cql_lagrange, 'cql_target_action_gap' : cql_target_action_gap, 'cql_temp' : cql_temp, 'cql_max_target_backup' : cql_max_target_backup, 'cql_clip_diff_min' : cql_clip_diff_min, 'cql_clip_diff_max' : cql_clip_diff_max,})

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