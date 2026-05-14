import numpy as np

#earth constants
MU_EARTH = 3.986004418e14      # m^3/s^2
R_EARTH = 6378137.0            # m
OMEGA_EARTH = 7.2921159e-5     # rad/s
J2 = 1.08262668e-3             # -

G0 = 9.80665                   # m/s^2

#atmosphere constants
RHO0 = 1.225                   # kg/m^3
SCALE_HEIGHT = 8500.0          # m
ATMOSPHERE_CUTOFF = 120000.0   # m

#time helpers
SECONDS_PER_DAY = 86400.0

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi