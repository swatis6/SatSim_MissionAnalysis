import numpy as np

from satsim.environment.earth import Environment
from satsim.environment.sun import AU_KM, in_shadow, sun_position_eci
from satsim.types import StateVector, Vec3

# https://en.wikipedia.org/w/index.php?title=Solar_constant&oldid=1340822418
_SOLAR_FLUX = np.float64(1361.0)  # W/m^2 at 1 AU
_C_LIGHT = np.float64(299_792_458.0)  # m/s


def accel_2body(state: StateVector, t: np.float64, env: Environment) -> Vec3:
    # pulls toward Earth center; a = -mu/r^3 * r (r^3 folds in the unit vector)
    r_mag = np.linalg.norm(state.r)
    return -env.mu / r_mag**3 * state.r


def accel_j2(state: StateVector, t: np.float64, env: Environment) -> Vec3:
    # Earth bulges at the equator; J2 corrects for the asymmetry
    r = state.r
    r_mag = np.linalg.norm(r)
    x, y, z = r
    factor = -1.5 * env.J2 * env.mu * env.R_e**2 / r_mag**5
    z2 = (z / r_mag) ** 2  # sin^2 of latitude
    return np.array(
        [
            factor * x * (1.0 - 5.0 * z2),
            factor * y * (1.0 - 5.0 * z2),
            factor * z * (3.0 - 5.0 * z2),
        ]
    )


def gravity(state: StateVector, t: np.float64, env: Environment) -> Vec3:
    # total gravity: two-body + J2 perturbation
    return accel_2body(state, t, env) + accel_j2(state, t, env)


def make_accel_srp(mass: float, area: float, reflectivity: float):
    """
    create an SRP force function for a spacecraft with cannonball model

    args:
        mass: spacecraft mass, kg
        area: cross-section area, m^2
        reflectivity: radiation pressure coefficient (1.0 = absorber, 2.0 = perfect reflector)
    """
    # cannonball SRP model: a = (S/c) * C_R * (A/m) * (AU/d)^2 * s_hat
    # https://en.wikipedia.org/wiki/Radiation_pressure — P = S/c for absorbing surface
    # https://ntrs.nasa.gov/citations/20205005240 — cannonball: a = P * C_R * A/m
    # S/c: W/m^2 / (m/s) = kg/s^3 / (m/s) = kg/(m*s^2) = Pa
    # A/m: Pa * m^2/kg = (N/m^2)*m^2/kg = N/kg = m/s^2
    # 1e-3: m/s^2 -> km/s^2
    pressure_accel = (_SOLAR_FLUX / _C_LIGHT) * reflectivity * (area / mass) * 1e-3

    def accel_srp(state: StateVector, t: np.float64, env: Environment) -> Vec3:
        sun_pos = sun_position_eci(env.epoch, t)
        if in_shadow(state.r, sun_pos, env.R_e):
            return np.zeros(3)

        sat_to_sun = sun_pos - state.r
        distance = np.linalg.norm(sat_to_sun)

        # force pushes satellite away from sun
        return -pressure_accel * (AU_KM / distance) ** 2 * (sat_to_sun / distance)

    return accel_srp
