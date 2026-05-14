import numpy as np

def gravitational_acceleration(spacecraft, environment):
    return environment.gravity.acceleration(spacecraft.state.r)

def drag_acceleration(spacecraft, environment):
    #a_drag = -0.5 * rho * |v_rel| * v_rel / BC
    #BC = m / (Cd * A) --> ballistic coefficient

    r = spacecraft.state.r
    v = spacecraft.state.v

    rho = environment.atmosphere.density(r)
    if rho < 1e-15:                # negligible above 1000 km
        return np.zeros(3)

    # v relative to atm. later, subtract omega_earth × r to acct for earth rotation.
    v_rel = v
    v_rel_norm = np.linalg.norm(v_rel)

    BC = spacecraft.mass_props.ballistic_coefficient
    return -0.5 * rho * v_rel_norm * v_rel / BC

def total_acceleration(spacecraft, environment):
    """
    Sum of all accelerations acting on the spacecraft.
    This is the function the integrator calls.
    """
    a = gravitational_acceleration(spacecraft, environment)
    a += drag_acceleration(spacecraft, environment)
    return a