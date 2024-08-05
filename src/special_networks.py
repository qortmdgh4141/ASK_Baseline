from jaxrl_m.dataset import Dataset
from jaxrl_m.typing import *
from jaxrl_m.networks import *
import jax

class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x
    
class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(observations)

class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final, activations=nn.gelu)(observations)
    
class RelativeRepresentation(nn.Module):
    rep_dim: int = 256
    hidden_dims: tuple = (256, 256)
    module: nn.Module = None
    visual: bool = False
    layer_norm: bool = False
    rep_type: str = 'concat' 
    bottleneck: bool = True  # Meaning that we're using this representation for high-level actions

    @nn.compact
    def __call__(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = jax.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-6, targets, bases)
            elif self.rep_type == 'concat':
                inputs = jax.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases)
            else:
                raise NotImplementedError

        if self.visual:
            inputs = self.module()(inputs)
        if self.layer_norm:
            rep = LayerNormMLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)
        else:
            rep = MLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)

        if self.bottleneck:
            rep = rep / jnp.linalg.norm(rep, axis=-1, keepdims=True) * jnp.sqrt(self.rep_dim)

        return rep

class MonolithicVF(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    # rep_dim: int = None
    obs_rep: int = 0

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class((*self.hidden_dims, 1), activate_final=False)

    def __call__(self, observations, goals=None, info=False):
        phi = observations
        psi = goals

        v1, v2 = self.value_net(jnp.concatenate([phi, psi], axis=-1)).squeeze(-1)

        if info:
            return {
                'v': (v1 + v2) / 2,
            }
        return v1, v2

class MonolithicQF(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    # rep_dim: int = None
    obs_rep: int = 0
    bilinear: int = 0
    
    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        if self.bilinear:
            self.s_a = repr_class((*self.hidden_dims, self.bilinear), activate_final=False)
            self.s_g = repr_class((*self.hidden_dims, self.bilinear), activate_final=False)
            
        else:
            self.q_net = repr_class((*self.hidden_dims, 1), activate_final=False)
    def __call__(self, observations, actions, subgoals=None, goals=None, info=False):

        if self.bilinear:
            if goals is not None:
                s_a = self.s_a(jnp.concatenate([observations, actions], axis=-1))
                s_g = self.s_g(jnp.concatenate([observations, subgoals, goals], axis=-1))
                
            else:
                s_a = self.s_a(jnp.concatenate([observations, actions], axis=-1))
                s_g = self.s_g(jnp.concatenate([observations, subgoals], axis=-1))
            
            einsum_str = 'ijk,ijk->ij' if len(s_a.shape) == 3 else 'ijkl,ijkl->ijk'
            q1, q2 = jnp.einsum(einsum_str, s_a, s_g)
        
        else:
            
            if goals is not None:
                q1, q2 = self.q_net(jnp.concatenate([observations, actions, subgoals, goals], axis=-1)).squeeze(-1)
            else:
                q1, q2 = self.q_net(jnp.concatenate([observations, actions, subgoals], axis=-1)).squeeze(-1)
            

        if info:
            return {
                'q': (q1 + q2) / 2,
            }
        return q1, q2

class Scalar(nn.Module):

    def setup(self):
        self.value = self.param('value', nn.initializers.ones, 1)
        
    def __call__(self):
        return self.value
    
class HILP_GoalConditionedPhiValue(nn.Module):
    hidden_dims: tuple = (256, 256) # (512, 512, 512)
    readout_size: tuple = (256,)
    skill_dim: int = 2 # 32
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None # False
    obs_dim: int = 0

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        phi = repr_class((*self.hidden_dims, self.skill_dim), activate_final=False, ensemble=self.ensemble)
        
        decoder = repr_class((*self.hidden_dims, self.obs_dim), activate_final=False)
        # If HILP on visual-kitchen-partial impala_small; else None # 
        if self.encoder is not None: 
            phi = nn.Sequential([self.encoder(), phi])
        self.phi = phi
        self.decoder = decoder

    def get_phi(self, observations):
        return self.phi(observations)[0]  # Use the first vf

    def __call__(self, observations, goals=None, info=False):
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)
        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
        
        if self.obs_dim:
            recon_obs = self.decoder(phi_s)
            recon_goals = self.decoder(phi_g)
            return v, (recon_obs, recon_goals)
        
        return v
    
class Vae_Encoder(nn.Module):
    hidden_dim: Sequence[int]
    rep_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    layer_norm : bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        for i, size in enumerate(self.hidden_dim):
            x = nn.Dense(size, kernel_init = self.kernel_init)(x)
            x = self.activations(x)
        mean = nn.Dense(self.rep_dim, kernel_init = self.kernel_init)(x)
        log_stddev = nn.Dense(self.rep_dim, kernel_init = self.kernel_init)(x)
        stddev = jnp.exp(log_stddev)
        z = mean
        return z, mean, stddev

class Vae_Decoder(nn.Module):
    hidden_dim: Sequence[int]
    output_shape: Sequence[int] 
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    layer_norm : bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dim):
            x = nn.Dense(size, kernel_init = self.kernel_init)(x)
            x = self.activations(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        output = nn.Dense(self.output_shape, kernel_init = self.kernel_init)(x)
        return output

def get_rep(
        encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)

class HierarchicalActorCritic(nn.Module):
    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]
    flag: Any = None
# ---------------------------------------------------------------------------------------------------------------    
    # HIQL
    def value(self, observations, goals, **kwargs):
        if self.flag.use_rep in ["hiql_goal_encoder", "vae_encoder"]:
            goals = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['value'](observations, goals, **kwargs)

    def target_value(self, observations, goals, **kwargs):
        if self.flag.use_rep in ["hiql_goal_encoder", "vae_encoder"]:
            goals = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['target_value'](observations, goals, **kwargs)

    def actor(self, observations, goals, subgoals=None, low_dim_goals=False, state_rep_grad=True, goal_rep_grad=True, **kwargs):
    
        goal_reps = goals
        if low_dim_goals:
            goal_reps = goals
        else:
            if self.flag.use_rep in ["hiql_goal_encoder", "vae_encoder"] :
                goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
            if not goal_rep_grad: # goal_rep_grad=False: low actor때는 업데이트 안함.
                goal_reps = jax.lax.stop_gradient(goal_reps)
        return self.networks['actor'](jnp.concatenate([observations, goal_reps], axis=-1), **kwargs)

    def high_actor(self, observations, goals, state_rep_grad=True, goal_rep_grad=True, **kwargs):
        goal_reps = goals
        return self.networks['high_actor'](jnp.concatenate([observations, goal_reps], axis=-1), **kwargs)

    def value_goal_encoder(self, targets, bases, **kwargs):
        return get_rep(self.encoders['value_goal'], targets=targets, bases=bases)

    def policy_goal_encoder(self, targets, goals, **kwargs):
        assert not self.use_waypoints
        return get_rep(self.encoders['policy_goal'], targets=targets, bases=goals)
# ---------------------------------------------------------------------------------------------------------------
    # HILP
    def hilp_value(self, observations, goals=None, **kwargs):
        return self.networks['hilp_value'](observations, goals, **kwargs)

    def hilp_target_value(self, observations, goals=None, **kwargs):
        return self.networks['hilp_target_value'](observations, goals, **kwargs)
    
    def hilp_phi(self, observations):
        return self.networks['hilp_value'].get_phi(observations)
# ---------------------------------------------------------------------------------------------------------------
    # VAE
    def vae_state_encoder(self, targets, **kwargs):
        return get_rep(self.encoders['vae_state_encoder'], targets=targets)
    
    def vae_state_decoder(self, targets, **kwargs):
        return get_rep(self.encoders['vae_state_decoder'], targets=targets)
# ---------------------------------------------------------------------------------------------------------------
    # 네트워크 초기화
    def __call__(self, observations=None, goals=None, subgoals=None, latent=None):
        if self.flag.use_rep == "hiql_goal_encoder":
            rets = {
                'value': self.value(observations, goals),
                'target_value': self.target_value(observations, goals),
                'actor': self.actor(observations, goals),
                'high_actor': self.high_actor(observations, goals),
            }
        elif self.flag.use_rep =="hilp_subgoal_encoder":
            rets = {            
                'hilp_value': self.hilp_value(observations, goals), 
                'hilp_target_value': self.hilp_target_value(observations, goals),
                'value': self.value(observations, goals[:, :self.flag.hilp_skill_dim]),
                'target_value': self.target_value(observations, goals[:, :self.flag.hilp_skill_dim]),
                'actor': self.actor(observations, goals[:, :self.flag.hilp_skill_dim]),
                'high_actor': self.high_actor(observations, goals[:, :self.flag.hilp_skill_dim]),
            }
        elif self.flag.use_rep == "hilp_encoder":
            rets = {            
                'hilp_value': self.hilp_value(observations, goals), 
                'hilp_target_value': self.hilp_target_value(observations, goals),
                'value': self.value(observations[:, :self.flag.hilp_skill_dim], goals[:, :self.flag.hilp_skill_dim]),
                'target_value': self.target_value(observations[:, :self.flag.hilp_skill_dim], goals[:, :self.flag.hilp_skill_dim]),
                'actor': self.actor(observations[:, :self.flag.hilp_skill_dim], goals[:, :self.flag.hilp_skill_dim]),
                'high_actor': self.high_actor(observations[:, :self.flag.hilp_skill_dim], goals[:, :self.flag.hilp_skill_dim]),
            }
        elif self.flag.use_rep == "vae_encoder":
            base_observations = observations
            observations = observations[:, :self.flag.vae_encoder_dim]
            goals = goals[:, :self.flag.vae_encoder_dim]
            latent = observations[:, :self.flag.vae_encoder_dim]
            rets = {
                'vae_state_encoder' : self.vae_state_encoder(base_observations), 
                'vae_state_decoder' : self.vae_state_decoder(latent), 
                'value_goal' : self.value_goal_encoder(observations, goals), 
                'value': self.value(observations, goals),
                'target_value': self.target_value(observations, goals),
                'actor': self.actor(observations, goals),
                'high_actor': self.high_actor(observations, goals),
            }
        elif self.flag.final_goal:
            rets = {
                'hilp_value': self.hilp_value(observations, goals), 
                'hilp_target_value': self.hilp_target_value(observations, goals),
                'value': self.value(observations, goals),
                'target_value': self.target_value(observations, goals),
                'actor': self.actor(observations, goals, subgoals=subgoals),
                'high_actor': self.high_actor(observations, goals),
            }
        elif self.flag.high_action_in_hilp:
            rets = {
                'hilp_value': self.hilp_value(observations, goals), 
                'hilp_target_value': self.hilp_target_value(observations, goals),
                'value': self.value(observations, goals),
                'target_value': self.target_value(observations, goals),
                'actor': self.actor(observations, subgoals),
                'high_actor': self.high_actor(observations, goals),
            }
        else:
            rets = {
                'hilp_value': self.hilp_value(observations, goals), 
                'hilp_target_value': self.hilp_target_value(observations, goals),
                'value': self.value(observations, goals),
                'target_value': self.target_value(observations, goals),
                'actor': self.actor(observations, goals),
                'high_actor': self.high_actor(observations, goals),
            }
            
        return rets


class HierarchicalActorCritic_HCQL(nn.Module):
    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]
    flag: Any = None
# ---------------------------------------------------------------------------------------------------------------    
    # HCQL
    def qf(self, observations, actions, subgoals, goals, **kwargs):
        # if self.flag.use_rep in ["hiql_goal_encoder", "vae_encoder"]:
        state_reps = get_rep(self.encoders['qf_state'], targets=observations)
        goal_reps = get_rep(self.encoders['qf_goal'], targets=goals, bases=observations)
        if self.flag.final_goal:
            return self.networks['qf'](state_reps, actions, subgoals, goal_reps, **kwargs)
        else:
            return self.networks['qf'](state_reps, actions, goal_reps, **kwargs)

    def target_qf(self, observations, actions, subgoals, goals, **kwargs):
        # if self.flag.use_rep in ["hiql_goal_encoder", "vae_encoder"]:
        state_reps = get_rep(self.encoders['qf_state'], targets=observations)
        goal_reps = get_rep(self.encoders['qf_goal'], targets=goals, bases=observations)
        if self.flag.final_goal:
            return self.networks['target_qf'](state_reps, actions, subgoals, goal_reps, **kwargs)
        else:
            return self.networks['target_qf'](state_reps, actions, goal_reps, **kwargs)
    
    # hierarcy qf function
    def high_qf(self, observations, actions, goals, **kwargs):
        # if self.flag.use_rep in ["hiql_goal_encoder", "vae_encoder"]:
        state_rep = get_rep(self.encoders['high_qf_state'], targets=observations)
        goal_reps = get_rep(self.encoders['high_qf_goal'], targets=goals, bases=observations)
        return self.networks['high_qf'](state_rep, actions, goal_reps, **kwargs)

    def high_target_qf(self, observations, actions, goals, **kwargs):
        # if self.flag.use_rep in ["hiql_goal_encoder", "vae_encoder"]:
        state_rep = get_rep(self.encoders['high_qf_state'], targets=observations)
        goal_reps = get_rep(self.encoders['high_qf_goal'], targets=goals, bases=observations)
        return self.networks['high_target_qf'](state_rep, actions, goal_reps, **kwargs)

    def actor(self, observations, subgoals, goals, low_dim_goals=False, state_rep_grad=True, goal_rep_grad=True, **kwargs):
        
        # assert goals.shape[-1] == 3, 'goals dim error'
        
        state_reps = get_rep(self.encoders['encoder'], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)
        
        if low_dim_goals:
            goal_reps = goals
        else:
            # if self.flag.use_rep in ["hiql_goal_encoder", "vae_encoder"] :
            goal_reps = get_rep(self.encoders['encoder'], targets=goals)
            if not goal_rep_grad: # goal_rep_grad=False: low actor때는 업데이트 안함.
                goal_reps = jax.lax.stop_gradient(goal_reps)
        if self.flag.final_goal:
            return self.networks['actor'](jnp.concatenate([state_reps, subgoals, goal_reps], axis=-1), **kwargs)
        else:
            return self.networks['actor'](jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs)

    def high_actor(self, observations, goals, state_rep_grad=True, goal_rep_grad=True, **kwargs):
        state_reps = get_rep(self.encoders['encoder'], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)
        
        goal_reps = get_rep(self.encoders['encoder'], targets=goals)
        if not goal_rep_grad:
            goal_reps = jax.lax.stop_gradient(goal_reps)
        
        return self.networks['high_actor'](jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs)

    def log_alpha(self, **kwargs):
        return self.networks['log_alpha']()[0]

    def high_log_alpha(self, **kwargs):
        return self.networks['high_log_alpha']()[0]

    def log_alpha_prime(self, **kwargs):
        return self.networks['log_alpha_prime']()[0]

    def high_log_alpha_prime(self, **kwargs):
        return self.networks['high_log_alpha_prime']()[0]
    
    def qf_goal_encoder(self, targets, bases, **kwargs):
        return get_rep(self.encoders['qf_goal'], targets=targets, bases=bases)
    
    def high_qf_goal_encoder(self, targets, bases, **kwargs):
        return get_rep(self.encoders['high_qf_goal'], targets=targets, bases=bases)

    def policy_goal_encoder(self, targets, goals, **kwargs):
        assert not self.use_waypoints
        return get_rep(self.encoders['policy_goal'], targets=targets, bases=goals)
# ---------------------------------------------------------------------------------------------------------------
    # HILP
    def hilp_value(self, observations, goals=None, **kwargs):
        return self.networks['hilp_value'](observations, goals, **kwargs)

    def hilp_target_value(self, observations, goals=None, **kwargs):
        return self.networks['hilp_target_value'](observations, goals, **kwargs)
    
    def hilp_phi(self, observations):
        return self.networks['hilp_value'].get_phi(observations)
# ---------------------------------------------------------------------------------------------------------------
    # VAE
    def vae_state_encoder(self, targets, **kwargs):
        return get_rep(self.encoders['vae_state_encoder'], targets=targets)
    
    def vae_state_decoder(self, targets, **kwargs):
        return get_rep(self.encoders['vae_state_decoder'], targets=targets)
# ---------------------------------------------------------------------------------------------------------------
    # 네트워크 초기화
    def __call__(self, observations=None, actions=None, goals=None, latent=None):        
        # if self.flag.use_rep == "hiql_goal_encoder":
        #     rets = {
        #         'high_qf_goal_encoder' : self.high_qf_goal_encoder(observations, goals), 
        #         'qf_goal_encoder' : self.qf_goal_encoder(observations, goals), 
        #         'qf': self.qf(observations, goals),
        #         'target_qf': self.target_qf(observations, goals),
        #         'actor': self.actor(observations, goals),
        #         'high_actor': self.high_actor(observations, goals),
        #     }
        # elif self.flag.use_rep =="hilp_subgoal_encoder":
        #     rets = {            
        #         'hilp_value': self.hilp_value(observations, goals), 
        #         'hilp_target_value': self.hilp_target_value(observations, goals),
        #         'value': self.value(observations, goals[:, :self.flag.hilp_skill_dim]),
        #         'target_value': self.target_value(observations, goals[:, :self.flag.hilp_skill_dim]),
        #         'actor': self.actor(observations, goals[:, :self.flag.hilp_skill_dim]),
        #         'high_actor': self.high_actor(observations, goals[:, :self.flag.hilp_skill_dim]),
        #     }
        # elif self.flag.use_rep == "hilp_encoder":
        #     rets = {            
        #         'hilp_value': self.hilp_value(observations, goals), 
        #         'hilp_target_value': self.hilp_target_value(observations, goals),
        #         'value': self.value(observations[:, :self.flag.hilp_skill_dim], goals[:, :self.flag.hilp_skill_dim]),
        #         'target_value': self.target_value(observations[:, :self.flag.hilp_skill_dim], goals[:, :self.flag.hilp_skill_dim]),
        #         'actor': self.actor(observations[:, :self.flag.hilp_skill_dim], goals[:, :self.flag.hilp_skill_dim]),
        #         'high_actor': self.high_actor(observations[:, :self.flag.hilp_skill_dim], goals[:, :self.flag.hilp_skill_dim]),
        #     }
        # elif self.flag.use_rep == "vae_encoder":
        #     base_observations = observations
        #     observations = observations[:, :self.flag.vae_encoder_dim]
        #     goals = goals[:, :self.flag.vae_encoder_dim]
        #     latent = observations[:, :self.flag.vae_encoder_dim]
        #     rets = {
        #         'vae_state_encoder' : self.vae_state_encoder(base_observations), 
        #         'vae_state_decoder' : self.vae_state_decoder(latent), 
        #         'value_goal' : self.value_goal_encoder(observations, goals), 
        #         'value': self.value(observations, goals),
        #         'target_value': self.target_value(observations, goals),
        #         'actor': self.actor(observations, goals),
        #         'high_actor': self.high_actor(observations, goals),
        #     }
        
        # elif 'Fetch' in self.flag.env_name:
        #     subgoal = goals if self.flag.small_subgoal_space else observations
        #     rets = {
        #         'qf': self.qf(observations, actions, goals),
        #         'target_qf': self.target_qf(observations, actions, goals),
        #         'actor': self.actor(observations, goals),
        #         'high_actor': self.high_actor(observations, goals),
        #         'log_alpha': self.log_alpha()
        #     }
        
        low_subgoals = jnp.zeros((1,self.flag.hilp_skill_dim*2)) if self.flag.high_action_in_hilp else goals
        high_subgoals = jnp.zeros((1,self.flag.hilp_skill_dim)) if self.flag.high_action_in_hilp else goals
            
        
        if self.flag.final_goal:
            rets = {
                
                'hilp_value': self.hilp_value(observations, goals), 
                'hilp_target_value': self.hilp_target_value(observations, goals),
                'high_qf' : self.high_qf(observations, high_subgoals, goals), 
                'high_target_qf' : self.high_target_qf(observations, high_subgoals, goals), 
                'qf': self.qf(observations, actions, low_subgoals, goals),
                'target_qf': self.target_qf(observations, actions, low_subgoals, goals),
                'actor': self.actor(observations, low_subgoals, goals),
                'high_actor': self.high_actor(observations, goals),
                'log_alpha' : self.log_alpha(),
                'high_log_alpha' : self.high_log_alpha(),
                'log_alpha_prime' : self.log_alpha_prime(),
                'high_log_alpha_prime' : self.high_log_alpha_prime(),
            }
        else:
            rets = {
            
            'hilp_value': self.hilp_value(observations, goals), 
            'hilp_target_value': self.hilp_target_value(observations, goals),
            'high_qf' : self.high_qf(observations, high_subgoals, goals), 
            'high_target_qf' : self.high_target_qf(observations, high_subgoals, goals), 
            'qf': self.qf(observations, actions, low_subgoals),
            'target_qf': self.target_qf(observations, actions, low_subgoals),
            'actor': self.actor(observations, low_subgoals),
            'high_actor': self.high_actor(observations, goals),
            'log_alpha' : self.log_alpha(),
            'high_log_alpha' : self.high_log_alpha(),
            'log_alpha_prime' : self.log_alpha_prime(),
            'high_log_alpha_prime' : self.high_log_alpha_prime(),
        }

                                
        return rets