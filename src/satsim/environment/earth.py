import numpy as np
from satsim.utilities.consts import MU_EARTH, R_EARTH, OMEGA_EARTH, J2
class EarthModel:
    def __init__(self):
        self.radius = R_EARTH
        self.omega = OMEGA_EARTH