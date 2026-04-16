from collections.abc import Callable

import numpy as np

from satsim.environment.earth import Environment
from satsim.types import StateVector, Vec3

# force fn: (state, t, env) -> acceleration
AccelFn = Callable[[StateVector, np.float64, Environment], Vec3]


def _deriv(
    state: StateVector, t: np.float64, accel_fn: AccelFn, env: Environment
) -> np.ndarray:
    # d/dt[pos, vel] = [vel, accel]
    a = accel_fn(state, t, env)
    return np.concatenate([state.v, a])


def _vec_to_state(arr: np.ndarray) -> StateVector:
    return StateVector(r=arr[:3], v=arr[3:])


def rk4_step(
    state: StateVector,
    t: np.float64,
    dt: np.float64,
    accel_fn: AccelFn,
    env: Environment,
) -> StateVector:
    # samples 4 derivative estimates and takes a weighted avg; more accurate than Euler
    arr = np.concatenate([state.r, state.v])
    k1 = _deriv(_vec_to_state(arr), t, accel_fn, env)
    k2 = _deriv(_vec_to_state(arr + 0.5 * dt * k1), t + dt / 2, accel_fn, env)
    k3 = _deriv(_vec_to_state(arr + 0.5 * dt * k2), t + dt / 2, accel_fn, env)
    k4 = _deriv(_vec_to_state(arr + dt * k3), t + dt, accel_fn, env)
    result = arr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _vec_to_state(result)


def propagate(
    state0: StateVector,
    t_span: np.float64,  # total simulation time, seconds
    dt: np.float64,  # step size, seconds
    force_fns: list[AccelFn],
    env: Environment,
    reentry_alt_km: np.float64 = np.float64(
        80.0
    ),  # stop if satellite drops below this altitude
) -> tuple[np.ndarray, list[StateVector]]:
    def combined_accel(state: StateVector, t: np.float64, env: Environment) -> Vec3:
        return sum((f(state, t, env) for f in force_fns), np.zeros(3))

    n = int(t_span / dt)
    times = np.empty(n + 1)
    states: list[StateVector] = []
    times[0] = 0.0
    states.append(state0)
    reentry_r = env.R_e + reentry_alt_km

    current = state0
    for i in range(n):
        t = times[i]
        current = rk4_step(current, t, dt, combined_accel, env)
        times[i + 1] = t + dt
        states.append(current)
        if np.linalg.norm(current.r) < reentry_r:
            return times[: i + 2], states

    return times, states
