import numpy as np
from satsim.utilities.consts import R_EARTH

class State:
    def __init__(self, r, v,
                 quaternion=None, omega=None):
        self.r = np.array(r, dtype=float)
        self.v = np.array(v, dtype=float)
        self.quaternion = np.array(quaternion if quaternion is not None
                                   else [0, 0, 0, 1], dtype=float)
        self.omega = np.array(omega if omega is not None
                              else [0, 0, 0], dtype=float)

    @property
    def altitude(self):
        return np.linalg.norm(self.r) - R_EARTH