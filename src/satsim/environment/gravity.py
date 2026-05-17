import numpy as np
from satsim.utilities.consts import MU_EARTH, R_EARTH, J2

class GravityModel:
    def __init__(self, config):
        self.use_j2 = config.get("use_j2", True)

    def acceleration(self, r):
        r_norm = np.linalg.norm(r)

        a_two_body = -MU_EARTH * r / r_norm**3

        if not self.use_j2:
            return a_two_body

        x, y, z = r
        factor = -1.5 * J2 * MU_EARTH * R_EARTH**2 / r_norm**5
        z_ratio_sq = 5.0 * (z / r_norm)**2

        a_j2 = factor * np.array([
            x * (1.0 - z_ratio_sq),
            y * (1.0 - z_ratio_sq),
            z * (3.0 - z_ratio_sq),
        ])

        return a_two_body + a_j2