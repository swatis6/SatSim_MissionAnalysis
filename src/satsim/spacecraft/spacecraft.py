from spacecraft.state import State
from spacecraft.massprops import MassProps
from spacecraft.statusflags import StatusFlags

class Spacecraft:
    def __init__(self, config):
        # Initial state from config
        initial = config.get("initial_state", {})
        self.state = State(
            r=initial.get("r", [6.778e6, 0, 0]),
            v=initial.get("v", [0, 7669.0, 0]),
        )
        self.mass_props = MassProps(config.get("mass_props", {}))
        self.flags = StatusFlags()
        self.name = config.get("name", "satellite")

        # subsytems for later
        self.adcs = None
        self.comms = None