import numpy as np
from satsim.dynamics.forces import total_acceleration

def rk4_step(spacecraft, environment, dt):
    #fixed step rk4 to start
    
    # initial state
    r0 = spacecraft.state.r.copy()
    v0 = spacecraft.state.v.copy()

    # k1: start
    a1 = total_acceleration(spacecraft, environment)
    k1_r = v0
    k1_v = a1

    # k2: midpoint, using k1
    spacecraft.state.r = r0 + 0.5 * dt * k1_r
    spacecraft.state.v = v0 + 0.5 * dt * k1_v
    a2 = total_acceleration(spacecraft, environment)
    k2_r = spacecraft.state.v
    k2_v = a2

    # k3: midpoint, using k2
    spacecraft.state.r = r0 + 0.5 * dt * k2_r
    spacecraft.state.v = v0 + 0.5 * dt * k2_v
    a3 = total_acceleration(spacecraft, environment)
    k3_r = spacecraft.state.v
    k3_v = a3

    # k4: end, using k3
    spacecraft.state.r = r0 + dt * k3_r
    spacecraft.state.v = v0 + dt * k3_v
    a4 = total_acceleration(spacecraft, environment)
    k4_r = spacecraft.state.v
    k4_v = a4

    # combine: state += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    spacecraft.state.r = r0 + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    spacecraft.state.v = v0 + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)