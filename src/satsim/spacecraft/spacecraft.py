from satsim.spacecraft.state import State
from satsim.spacecraft.massprops import MassProps
from satsim.spacecraft.statusflags import StatusFlags

class Spacecraft:
    def __init__(self, config):
        # Initial state from config
        initial = config.get("initial_state")
        self.state = State(
            r=initial.get("r"),
            v=initial.get("v"),
        )
        self.mass_props = MassProps(config.get("mass_props"))
        self.flags = StatusFlags()
        self.name = config.get("name", "satellite")

        # subsytems for later
        self.adcs = None
        self.comms = None