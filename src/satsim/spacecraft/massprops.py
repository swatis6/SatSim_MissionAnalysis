class MassProps:
    def __init__(self, config):
        self.mass         = config.get("mass")         # kg (3U cubesat ~4kg)
        self.cd           = config.get("cd")           # drag coefficient
        self.area         = config.get("area")        # m^2 (3U face area)
        # inertia tensor for attitude dynamics
        self.inertia      = config.get("inertia")

    @property
    def ballistic_coefficient(self):
        #BC = m / (Cd * A), units kg/m^2
        return self.mass / (self.cd * self.area)