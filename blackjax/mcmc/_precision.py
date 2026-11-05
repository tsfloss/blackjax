"""Opt-in float64 evaluation of HMC energies (fork-local hack).

Set the environment variable ``float=64`` to evaluate the kinetic energy,
the total energy ``H = -logdensity + K`` and the energy difference
``H0 - H1`` in float64, while the rest of the program stays float32.

The user's ``logdensity_fn`` must *return* a float64 scalar for this to be
useful (e.g. reduce inside ``jax.enable_x64()`` and don't cast back).

Precision flows like this:
    logdensity (f64) , K (f64)  ->  H (f64)  ->  dH = H0 - H1 (f64)  ->  cast to f32
Everything after the subtraction (exp, Bernoulli draw, NUTS log-weights,
acceptance statistics) is O(1) and stays float32, which keeps dtypes
consistent across lax.cond / while_loop carries.

The flag is read at trace time, so it must be set before the kernel is jitted.
"""
import os

import jax.numpy as jnp


def use_x64() -> bool:
    return os.environ.get("float") == "64"


def as_f64(x):
    """Cast to float64. Must be called inside ``jax.enable_x64()``."""
    return jnp.asarray(x, dtype=jnp.float64)
