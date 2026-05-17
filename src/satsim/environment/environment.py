from satsim.environment.gravity import GravityModel
from satsim.environment.atmosphere import AtmosphereModel
from satsim.environment.earth import EarthModel

class Environment:
    def __init__(self, config):
        self.gravity     = GravityModel(config.get("gravity"))
        self.atmosphere  = AtmosphereModel(config.get("atmosphere"))
        self.earth       = EarthModel()