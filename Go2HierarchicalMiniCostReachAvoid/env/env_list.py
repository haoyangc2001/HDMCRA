from .reach_avoid.pendulum_constraint import PendulumConstraint
from .reach_avoid.hopper_avoid_ceiling import HopperAvoidCeiling, HopperAvoidCeilingDeterministic
from .reach_avoid.wind_field import WindField
from .reach_avoid.half_cheetah_avoid import HalfCheetahAvoid

from .wrappers import TransformObservation

from functools import partial
import jax.numpy as jnp

def transform_observation(mean, variance, obs):
    return (obs - mean) / variance

def get_env(config):
    if config["EXP_NAME"] == 'PendulumConstraint':
        trans = partial(transform_observation, jnp.array([0., 0., 0., 400.]), jnp.array([1., 1., 1., 400.]))
        env = PendulumConstraint()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'HopperAvoidCeiling':
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[0].set(1.)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = HopperAvoidCeiling()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'HalfCheetahAvoid':
        vec1 = jnp.zeros(20, dtype=jnp.float32)
        vec1 = vec1.at[0].set(2.5)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(20, dtype=jnp.float32)
        vec2 = vec2.at[0].set(3.)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = HalfCheetahAvoid()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'WindField':
        vec1 = jnp.zeros(14, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(14, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        vec2 = vec2.at[0].set(3.)
        vec2 = vec2.at[1].set(3.)
        vec2 = vec2.at[2].set(2.)
        trans = partial(transform_observation, vec1, vec2)
        env = WindField()
        env = TransformObservation(env, trans)
    elif config["EXP_NAME"] == 'F16Avoid':
        from .reach_avoid.F16_avoid import F16Avoid
        vec1 = jnp.zeros(26, dtype=jnp.float32)
        vec1 = vec1.at[-1].set(400.)
        vec2 = jnp.ones(26, dtype=jnp.float32)
        vec2 = vec2.at[-1].set(400.)
        trans = partial(transform_observation, vec1, vec2)
        env = F16Avoid()
        env = TransformObservation(env, trans)
    else:
        raise Exception("No Given Environment")
    return env