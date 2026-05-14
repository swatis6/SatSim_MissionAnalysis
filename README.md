# SATSIM: Mission Analysis

A modular Python-based simulation environment for modeling CubeSat missions, including orbital dynamics, attitude determination and control (ADCS), communications systems, and more.

This project is designed as a learning and prototyping tool for spacecraft dynamics, guidance, navigation, and control, and mission architecture analysis.

---

## Features


---

## Architecture Overview

Structure:

- `analysis/` 
- - `adcsmetrics.py`
- - `commsmetrics.py`
- - `coverage.py`
- - `montecarlo.py`
- `config/`
- - `adcs.yaml`
- - `comms.yaml`
- - `configclass.py`
- - `environment.yaml`
- - `mission.yaml`
- - `spacecraft.yaml`
- `data/`
- - `logs/`
- - `outputs/`
- - `plots/`
- `dynamics/`
- - `attitudedynamics.py`
- - `forces.py`
- - `orbitdynamics.py`
- - `torques.py`
- `environment/`
- - `atmosphere.py`
- - `earth.py`
- - `environment.py`
- - `gravity.py`
- - `timeutils.py`
- `mission/` 
- - `deploymentscenarios.py`
- - `events.py`
- - `mission.py`
- - `timeline.py`
- `simulation/` 
- - `eventsystem.py`
- - `propogator.py`
- - `scheduler.py`
- - `simulator.pi`
- `spacecraft/` 
- - `massprops.py`
- - `spacecraft.py`
- - `state.py`
- - `statusflags.py`
- `subsystems/` 
- - `adcs/`
- - - `actuators.py`
- - - `adcs.py`
- - - `controller.py`
- - - `estimators.py`
- - - `sensors.py`
- - `comms/`
- - - `antenna.py`
- - - `channel.py`
- - - `comms.py`
- - - `groundstation.py`
- - - `linkbudget.py`
- - `eps/`
- `utilities/`
- - `consts.py`
- - `coords.py`
- - `math.py`
- - `noisemodels.py`
- `visualization`
- `main.py`

---

## How to Run

```bash
# install dependencies
uv sync

# run simulation
uv run python src/satsim/main.py

# run tests
uv run pytest
