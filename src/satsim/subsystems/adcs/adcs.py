from satsim.subsystems.adcs.controller import Controller
from satsim.subsystems.adcs.estimator import Estimator 
from satsim.subsystems.adcs.sensors import Sensors

class ADCS:
    def __init__(self):
        self.controller = Controller()
        self.estimator = Estimator()
        self.sensors = Sensors()

def compute_torque(self, state):
    attitude = self.estimator.estimate(state)
    return self.controller.compute(attitude)
