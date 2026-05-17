import numpy as np
import pytest
import yaml
import numpy as np
from satsim.utilities.consts import MU_EARTH

from satsim.simulation.simulator import Simulator
from satsim.spacecraft.spacecraft import Spacecraft
from satsim.environment.environment import Environment
from satsim.mission.mission import Mission

class TestPropagator:

    # Sets up inital state for tests where J2 & drag are off.
    @classmethod
    def setup_class(cls):
        cls.runSteps = 553
        cls.dt = 10  # s
        sc_config = {
            "name": "TesterSat",
            "initial_state": {
                "r": [6.778137e6, 0, 0],  # 400 km altitude
                "v": [0, 7672.6, 0]},  # v for cicular orbit at altitude
            "mass_props": {
                "mass": 4.0,
                "cd": 2.2,
                "area": 0.03}}
        enviromentCong = {
            "gravity": {
                "use_j2": False},
            "atmosphere": {
                "rho0": 0,
                "scale_height": 8500.0}}  # No J2 / No Drag
        cls.sim = Simulator(Spacecraft(sc_config), Environment(enviromentCong), Mission(""))
        cls.duration = cls.dt * cls.runSteps

    # Testing that the altitude has almost not changed during the sim
    def test_Altitude_Conservation(self):
        self.sim.run(self.duration, self.dt)
        alt = np.array(self.sim.history["altitude"]) / 1000.0
        assert abs(alt[-1] - alt[0]) < 1

    # Testing that the energy is almost hasn't changed during the sim
    def test_energy_conservation(self):
        self.sim.run(self.duration, self.dt)
        pos = np.array(self.sim.history["r"])
        posMag = np.linalg.norm(pos, axis=1)
        vel = np.array(self.sim.history["v"])
        velMag = np.linalg.norm(vel, axis=1)
        energy = (1 / 2) * velMag ** 2 - (MU_EARTH / posMag)
        assert abs(energy[-1] - energy[0]) / energy[0] == pytest.approx(0, abs=1e-8)

    # Testing that the magnitude and direction of the satelitte is the same at the beginning and end of sim
    def test_angular_momentum_conservation(self):
        self.sim.run(self.duration, self.dt)
        pos = np.array(self.sim.history["r"])
        vel = np.array(self.sim.history["v"])
        h = np.cross(pos, vel)  # angular momentum
        hCrossMag = np.linalg.norm(h, axis=1)
        assert abs(hCrossMag[-1] - hCrossMag[0]) / hCrossMag[0] < 1e-10  # magnitude

        hUnitVectorStart = h[0] / np.linalg.norm(h[0])
        hUnitVectorEnd = h[-1] / np.linalg.norm(h[-1])
        hCrossDot = np.dot(hUnitVectorStart, hUnitVectorEnd)
        assert hCrossDot == pytest.approx(1, abs=1e-10)  # direction
