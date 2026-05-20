import yaml
from pathlib import Path

from satsim.simulation.simulator import Simulator
from satsim.spacecraft.spacecraft import Spacecraft
from satsim.environment.environment import Environment
from satsim.mission.mission import Mission
from satsim.output.output import Output

CONFIG_DIR = Path(__file__).parent / "config"

def load_yaml(name):
    with open(CONFIG_DIR / name, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    sc_config      = load_yaml("spacecraft.yaml")
    env_config     = load_yaml("environment.yaml")
    mission_config = load_yaml("mission.yaml")
    output_config     = load_yaml("output.yaml")

    sc      = Spacecraft(sc_config)
    env     = Environment(env_config)
    mission = Mission(mission_config)

    sim = Simulator(sc, env, mission)

    output = Output(sim.history, output_config)

    dt    = mission_config.get("dt", 10.0)
    t_end = mission_config.get("duration", 54000.0)

    sim.run(t_end, dt)
    print(f"Simulation complete. {len(sim.history['t'])} steps recorded.")

    output.run()

    

if __name__ == "__main__":
    main()