import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simulation.simulator import Simulator
from spacecraft.spacecraft import Spacecraft
from environment.environment import Environment
from mission.mission import Mission

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

    dt    = mission_config.get("dt", 10.0)
    t_end = mission_config.get("duration", 54000.0)

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