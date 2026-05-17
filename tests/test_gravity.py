import numpy as np
import pytest

from satsim.environment.gravity import GravityModel, MU_EARTH

class TestGravity:
    def test_two_body_acceleration_magnitude(self):
        # at earth surface gravity should be 9.81 m/s^2.
        config = {"use_j2": False}
        gravity = GravityModel(config)
        r = np.array([6.378137e6, 0, 0])  # surface
        a = gravity.acceleration(r)
        assert abs(np.linalg.norm(a) - 9.81) == pytest.approx(0, abs=0.1)

    def test_two_body_acceleration_direction(self):
        config = {"use_j2": False}
        gravity = GravityModel(config)
        r = np.array([6.378137e6, 0, 0])
        a = gravity.acceleration(r)
        expected_direction = -r / np.linalg.norm(r)  # unit vector towards earth
        assert np.allclose(a / np.linalg.norm(a), expected_direction)

    def test_j2_acceleration_effect(self):
        config = {"use_j2": True}
        gravity = GravityModel(config)
        r = np.array([6.378137e6, 0, 0])
        a_j2 = gravity.acceleration(r)
        a_no_j2 = -MU_EARTH * r / np.linalg.norm(r) ** 3
        assert not np.allclose(a_j2, a_no_j2)  # should differ