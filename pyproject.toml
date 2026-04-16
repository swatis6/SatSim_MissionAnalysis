from typing import Protocol, runtime_checkable

import numpy as np

from satsim.environment.earth import Environment
from satsim.spacecraft import Spacecraft
from satsim.types import StateVector, Vec3


@runtime_checkable
class ForceModel(Protocol):
    def acceleration(
        self, state: StateVector, t: np.float64, env: Environment
    ) -> Vec3: ...


@runtime_checkable
class StopCondition(Protocol):
    def check(self, state: StateVector, t: np.float64) -> bool: ...


@runtime_checkable
class Event(Protocol):
    trigger_time: np.float64

    def trigger(self, spacecraft: Spacecraft, env: Environment) -> None: ...
