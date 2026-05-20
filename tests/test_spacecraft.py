import numpy as np
import pytest
import yaml
from satsim.spacecraft.spacecraft import Spacecraft
from satsim.spacecraft.spacecraft import State
from satsim.spacecraft.spacecraft import MassProps

@pytest.fixture
def scConfig():
    with open("src/satsim/config/spacecraft.yaml", "r") as f:
        return yaml.safe_load(f) or {}

# Tests that the yamml position and velocity are the correct data tpye and are assigned fields
# correctly within Spacecraft
def test_storage_of_position_and_velocoity(scConfig):
    sc = Spacecraft(scConfig)
    initialState = scConfig.get("initial_state")
    assert np.array_equal(sc.state.r, initialState.get("r")) is True
    assert np.array_equal(sc.state.v, initialState.get("v")) is True
    assert isinstance(sc.state.r, np.ndarray) is True
    assert isinstance(sc.state.v, np.ndarray) is True

# Tests that the altitude equation is calculating the correct results
def test_altitude_property_computation():
    state = State([6.778137e6, 0, 0], [0, 0, 0])
    alt = state.altitude / 1000  # km
    assert alt == 400

# Tests that the altitude calculation will change if the position of the Spacecraft has changed
def test_altitude_changes_based_on_position(scConfig):
    scConfig["initial_state"]["r"] = np.array([7.378e6, 0.0, 0.0])
    sc = Spacecraft(scConfig)
    alt = sc.state.altitude / 1000
    assert alt == pytest.approx(1000, abs = 0.2)  # Not exactly 1000, but 999.86ish

# Tests that the Ballistic coefficient computes correctly
def test_ballistic_computation(scConfig):
    scConfig["mass_props"]["mass"] = 4.0  # kg
    scConfig["mass_props"]["cd"] = 2.2
    scConfig["mass_props"]["area"] = 0.03  # m^2
    massProps = MassProps(scConfig.get("mass_props"))
    coef = massProps.ballistic_coefficient
    assert coef == pytest.approx(60.6, abs = 0.1)  # 60.6 kg/m^2

# Tests that the Spacecraft's name is assigned correctly
def test_spacecraft_name(scConfig):
    scConfig["name"] = "THE BEST SATELLITE KNOWN TO MAN"
    sc = Spacecraft(scConfig)
    assert sc.name == "THE BEST SATELLITE KNOWN TO MAN"

# Tests that if a field isn't set in the yaml the Spacecraft class will default to the correct values
def test_null_values():
    nullConfig = {}
    sc = Spacecraft(nullConfig)
    assert sc.name == "satellite"
    assert np.array_equal(sc.state.r, [6.778e6, 0, 0])
    assert np.array_equal(sc.state.v, [0, 7669.0, 0])
    assert sc.mass_props.mass == 4.0
    assert sc.mass_props.cd == 2.2
    assert sc.mass_props.area == 0.03
    assert sc.mass_props.inertia is None
