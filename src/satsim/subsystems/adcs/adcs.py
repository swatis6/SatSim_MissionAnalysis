from subsystems.adcs.controller import Controller
from subsystems.adcs.estimator import Estimator 
from subsystems.adcs.sensors import Sensors

class ADCS:
    def __init__(self):
        self.controller = Controller()
        self.estimator = Estimator()
        self.sensors = Sensors()

def compute_torque(self, state):
    attitude = self.estimator.estimate(state)
    return self.controller.compute(attitude)
