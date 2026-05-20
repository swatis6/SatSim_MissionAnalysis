import numpy as np
import pytest
import yaml
import numpy as np
from satsim.utilities.consts import MU_EARTH

from satsim.simulation.simulator import Simulator
from satsim.spacecraft.spacecraft import Spacecraft
from satsim.environment.environment import Environment
from satsim.mission.mission import Mission

def load_yaml(name):
    with open("src/satsim/config/" + name, "r") as f:
        return yaml.safe_load(f) or {}

@pytest.fixture
def dt():
    return 10

@pytest.fixture
def duration(dt):
    runSteps = 553
    return dt * runSteps

@pytest.fixture
def scConfig():
    scConfig = load_yaml("spacecraft.yaml")
    scConfig["initial_state"]["r"] = [6.778137e6, 0, 0]
    scConfig["initial_state"]["v"] = [0, 7672.6, 0]
    scConfig["mass_props"]["mass"] = 4.0
    scConfig["mass_props"]["cd"] = 2.2
    scConfig["mass_props"]["area"] = 0.03
    return scConfig

@pytest.fixture
def enviromentConfig():
    enviromentConfig = load_yaml("environment.yaml")
    enviromentConfig["gravity"]["use_j2"] = False
    enviromentConfig["atmosphere"]["rho0"] = 0  # No J2 / No Drag
    enviromentConfig["atmosphere"]["scale_height"] = 8500.0
    return enviromentConfig

@pytest.fixture
def sim(duration, dt, scConfig, enviromentConfig):
    sim = Simulator(Spacecraft(scConfig), Environment(enviromentConfig), Mission(""))
    sim.run(duration, dt)
    alt = np.array(sim.history["altitude"]) / 1000.0 #km
    pos = np.array(sim.history["r"])
    vel = np.array(sim.history["v"])
    return alt, pos, vel

# Testing that the altitude has almost not changed during the sim
def test_Altitude_Conservation(sim):
    alt, pos, vel = sim
    assert abs(alt[-1] - alt[0]) < 1

# Testing that the energy is almost hasn't changed during the sim
def test_energy_conservation(sim):
    alt, pos, vel = sim
    posMag = np.linalg.norm(pos, axis = 1)
    velMag = np.linalg.norm(vel, axis = 1)
    energy = (1 / 2) * velMag ** 2 - (MU_EARTH / posMag)
    assert abs(energy[-1] - energy[0]) / energy[0] == pytest.approx(0, abs = 1e-8)

# Testing that the magnitude and direction of the satelitte is the same at the beginning and end of sim
def test_angular_momentum_conservation(sim):
    alt, pos, vel = sim
    h = np.cross(pos, vel)  # angular momentum
    hCrossMag = np.linalg.norm(h, axis = 1)
    assert abs(hCrossMag[-1] - hCrossMag[0]) / hCrossMag[0] < 1e-10  # magnitude

    hUnitVectorStart = h[0] / np.linalg.norm(h[0])
    hUnitVectorEnd = h[-1] / np.linalg.norm(h[-1])
    hCrossDot = np.dot(hUnitVectorStart, hUnitVectorEnd)
    assert hCrossDot == pytest.approx(1, abs = 1e-10)  # direction
