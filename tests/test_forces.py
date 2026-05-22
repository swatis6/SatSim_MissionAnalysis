import numpy as np

from satsim.dynamics.forces import drag_acceleration, total_acceleration
from satsim.environment.environment import Environment
from satsim.spacecraft.spacecraft import Spacecraft
from satsim.utilities.consts import R_EARTH


def make_spacecraft(*, r, v, mass=4.0, cd=2.2, area=0.03):
    return Spacecraft(
        {
            "initial_state": {"r": list(r), "v": list(v)},
            "mass_props": {"mass": mass, "cd": cd, "area": area},
        }
    )


def make_environment(*, use_j2, rho0, scale_height=8500.0):
    return Environment(
        {
            "gravity": {"use_j2": use_j2},
            "atmosphere": {"rho0": rho0, "scale_height": scale_height},
        }
    )


def test_total_acceleration_gravity_only():
    spacecraft = make_spacecraft(r=np.array([6.878137e6, 0.0, 0.0]), v=np.zeros(3))
    environment = make_environment(use_j2=False, rho0=0.0)

    total = total_acceleration(spacecraft, environment)
    gravity = environment.gravity.acceleration(spacecraft.state.r)

    assert np.allclose(total, gravity)


def test_total_acceleration_gravity_plus_j2_when_drag_off():
    spacecraft = make_spacecraft(r=np.array([6.878137e6, 0.0, 1000.0]), v=np.zeros(3))
    environment_no_j2 = make_environment(use_j2=False, rho0=0.0)
    environment_with_j2 = make_environment(use_j2=True, rho0=0.0)

    total = total_acceleration(spacecraft, environment_with_j2)
    gravity_only = environment_no_j2.gravity.acceleration(spacecraft.state.r)
    gravity_with_j2 = environment_with_j2.gravity.acceleration(spacecraft.state.r)
    j2_contribution = gravity_with_j2 - gravity_only

    assert np.allclose(total, gravity_only + j2_contribution)


def test_total_acceleration_gravity_j2_drag():
    spacecraft = make_spacecraft(r=np.array([6.878137e6, 0.0, 1000.0]), v=np.array([100.0, 7700.0, 0.0]))
    environment_no_j2 = make_environment(use_j2=False, rho0=1.225)
    environment_with_j2 = make_environment(use_j2=True, rho0=1.225)

    total = total_acceleration(spacecraft, environment_with_j2)
    gravity_only = environment_no_j2.gravity.acceleration(spacecraft.state.r)
    gravity_with_j2 = environment_with_j2.gravity.acceleration(spacecraft.state.r)
    j2_contribution = gravity_with_j2 - gravity_only
    drag = drag_acceleration(spacecraft, environment_with_j2)

    assert np.allclose(total, gravity_only + j2_contribution + drag)


def test_drag_opposes_velocity_along_x():
    spacecraft = make_spacecraft(r=np.array([R_EARTH + 10000.0, 0.0, 0.0]), v=np.array([500.0, 0.0, 0.0]))
    environment = make_environment(use_j2=False, rho0=1.225)

    drag = drag_acceleration(spacecraft, environment)

    assert drag[0] < 0.0
    assert drag[1] == 0.0
    assert drag[2] == 0.0


def test_drag_magnitude_scales_with_velocity_squared():
    environment = make_environment(use_j2=False, rho0=1.225)
    r = np.array([R_EARTH + 10000.0, 0.0, 0.0])
    sc_fast = make_spacecraft(r=r, v=np.array([200.0, 0.0, 0.0]))
    sc_slow = make_spacecraft(r=r, v=np.array([100.0, 0.0, 0.0]))

    drag_fast = np.linalg.norm(drag_acceleration(sc_fast, environment))
    drag_slow = np.linalg.norm(drag_acceleration(sc_slow, environment))

    assert drag_fast / drag_slow == 4.0


def test_drag_scales_linearly_with_density():
    r = np.array([R_EARTH + 10000.0, 0.0, 0.0])
    v = np.array([150.0, 20.0, 0.0])
    spacecraft = make_spacecraft(r=r, v=v)
    environment_low = make_environment(use_j2=False, rho0=0.5)
    environment_high = make_environment(use_j2=False, rho0=1.0)

    drag_low = np.linalg.norm(drag_acceleration(spacecraft, environment_low))
    drag_high = np.linalg.norm(drag_acceleration(spacecraft, environment_high))

    assert drag_high / drag_low == 2.0


def test_drag_inverse_with_ballistic_coefficient():
    r = np.array([R_EARTH + 10000.0, 0.0, 0.0])
    v = np.array([150.0, 0.0, 0.0])
    environment = make_environment(use_j2=False, rho0=1.225)
    sc_mass_4 = make_spacecraft(r=r, v=v, mass=4.0, cd=2.2, area=0.03)
    sc_mass_8 = make_spacecraft(r=r, v=v, mass=8.0, cd=2.2, area=0.03)

    drag_mass_4 = np.linalg.norm(drag_acceleration(sc_mass_4, environment))
    drag_mass_8 = np.linalg.norm(drag_acceleration(sc_mass_8, environment))

    assert drag_mass_4 / drag_mass_8 == 2.0
