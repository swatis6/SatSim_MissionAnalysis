from satsim.subsystems.comms.antenna import Antenna
from satsim.subsystems.comms.channel import Channel

class Comms:
    def __init__(self):
        self.antenna = Antenna()
        self.channel = Channel()

def update(self, state):
    if self.channel.is_visible(state):
        self.channel.transmit()