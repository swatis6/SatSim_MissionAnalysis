import numpy as np
from satsim.utilities.consts import R_EARTH

class AtmosphereModel:
    def __init__(self, config):
        # exponential atmosphere
        self.rho0          = config.get("rho0", 1.225)   
        self.scale_height  = config.get("scale_height", 8500.0)  

    def density(self, r):
        altitude = np.linalg.norm(r) - R_EARTH
        if altitude < 0:
            return self.rho0  #below sea level clamp
        return self.rho0 * np.exp(-altitude / self.scale_height)