import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from satsim.simulation.simulator import Simulator
from satsim.spacecraft.spacecraft import Spacecraft
from satsim.environment.environment import Environment
from satsim.mission.mission import Mission

CONFIG_DIR = Path(__file__).parent / "config"

def load_yaml(name):
    with open(CONFIG_DIR / name, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    sc_config      = load_yaml("spacecraft.yaml")
    env_config     = load_yaml("environment.yaml")
    mission_config = load_yaml("mission.yaml")

    sc      = Spacecraft(sc_config)
    env     = Environment(env_config)
    mission = Mission(mission_config)

    sim = Simulator(sc, env, mission)

    dt    = mission_config.get("dt")
    t_end = mission_config.get("duration")

    sim.run(t_end, dt)
    print(f"Simulation complete. {len(sim.history['t'])} steps recorded.")

    t = np.array(sim.history["t"]) / 60.0
    alt = np.array(sim.history["altitude"]) / 1000.0

    plt.figure()
    plt.plot(t, alt)
    plt.xlabel("Time (min)")
    plt.ylabel("Altitude (km)")
    plt.title("Orbit altitude vs time")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()