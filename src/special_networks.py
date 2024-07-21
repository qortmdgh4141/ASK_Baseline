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

class HILP_GoalConditionedPhiValue(nn.Module):
    hidden_dims: tuple = (256, 256) # (512, 512, 512)
    readout_size: tuple = (256,)
    skill_dim: int = 2 # 32
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None # False

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        phi = repr_class((*self.hidden_dims, self.skill_dim), activate_final=False, ensemble=self.ensemble)
        # If HILP on visual-kitchen-partial impala_small; else None # 
        if self.encoder is not None: 
            phi = nn.Sequential([self.encoder(), phi])
        self.phi = phi

    def get_phi(self, observations):
        return self.phi(observations)[0]  # Use the first vf

    def __call__(self, observations, goals=None, info=False):
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)
        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
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
