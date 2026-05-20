import numpy as np
from satsim.simulation.propagator import rk4_step

class Simulator:
    def __init__(self, spacecraft, environment, mission):
        self.sc = spacecraft
        self.env = environment
        self.mission = mission
        self.time = 0.0
        self.history = {"t": [], "r": [], "v": [], "altitude": []}

    def _record(self):
        self.history["t"].append(self.time)
        self.history["r"].append(self.sc.state.r.copy())
        self.history["v"].append(self.sc.state.v.copy())
        self.history["altitude"].append(self.sc.state.altitude)

    def step(self, dt):
        rk4_step(self.sc, self.env, dt)
        self.time += dt

        if self.sc.state.altitude < 100e3:
            self.sc.flags.decaying = True

        self._record()

    def run(self, duration, dt):
        n_steps = int(duration / dt)
        self._record()
        for _ in range(n_steps):
            if self.sc.flags.decaying:
                print(f"Decayed at t={self.time:.1f} s")
                break
            self.step(dt)
        
        for key in self.history:
            self.history[key] = np.array(self.history[key])
            