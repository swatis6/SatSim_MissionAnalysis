class MassProps:
    def __init__(self, config):
        self.mass         = config.get("mass", 4.0)         # kg (3U cubesat ~4kg)
        self.cd           = config.get("cd", 2.2)           # drag coefficient
        self.area         = config.get("area", 0.03)        # m^2 (3U face area)
        # inertia tensor for attitude dynamics
        self.inertia      = config.get("inertia", None)

    @property
    def ballistic_coefficient(self):
        #BC = m / (Cd * A), units kg/m^2
        return self.mass / (self.cd * self.area)