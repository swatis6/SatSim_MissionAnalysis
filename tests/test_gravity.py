import numpy as np
import pytest

from satsim.environment.gravity import GravityModel, MU_EARTH

@pytest.fixture
def r():
    return np.array([6.378137e6, 0, 0])

@pytest.fixture
def noJ2Accel(r):
    config = {"use_j2": False}
    gravity = GravityModel(config)
    return gravity.acceleration(r)

@pytest.fixture
def withJ2Accel(r):
    config = {"use_j2": True}
    gravity = GravityModel(config)
    return gravity.acceleration(r)

def test_two_body_acceleration_magnitude(noJ2Accel):
    # at earth surface gravity should be 9.81 m/s^2.
    assert abs(np.linalg.norm(noJ2Accel) - 9.81) == pytest.approx(0, abs = 0.1)

def test_two_body_acceleration_direction(noJ2Accel, r):
    expected_direction = -r / np.linalg.norm(r)  # unit vector towards earth
    assert np.allclose(noJ2Accel / np.linalg.norm(noJ2Accel), expected_direction)

def test_j2_acceleration_effect(withJ2Accel, r):
    a_no_j2 = -MU_EARTH * r / np.linalg.norm(r) ** 3
    assert not np.allclose(withJ2Accel, a_no_j2)  # should differ
