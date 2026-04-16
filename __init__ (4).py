from bisect import bisect_right
from dataclasses import dataclass, field
from math import exp

import astropy.time
import numpy as np

# piecewise exponential atmosphere - https://www.sciencedirect.com/science/article/pii/0032063372900991
# (base_alt_km, density_kg_m3, scale_height_km)
_ATMO = (
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
)
_ATMO_ALTS = tuple(row[0] for row in _ATMO)


@dataclass
class Environment:
    # WGS84 (standardized) constants - km, rad, s
    # https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84
    mu: np.float64 = np.float64(398600.4418)  # gravitational parameter, km^3/s^2
    R_e: np.float64 = np.float64(6378.137)  # equatorial radius, km
    J2: np.float64 = np.float64(1.08262982131e-3)  # second zonal harmonic
    f: np.float64 = np.float64(1.0 / 298.257223563)  # flattening
    omega_e: np.float64 = np.float64(7.2921150e-5)  # Earth rotation rate, rad/s

    # simulation clock
    epoch: astropy.time.Time = field(
        default_factory=lambda: astropy.time.Time("J2000", scale="tt")
    )
    t: np.float64 = np.float64(0.0)  # elapsed seconds since epoch
    dt: np.float64 = np.float64(10.0)  # default step size, s

    @property
    def R_p(self):
        # polar radius, km
        return self.R_e * (1.0 - self.f)

    def density(self, h_km) -> np.float64:
        # kg/m^3
        if h_km > 1000.0:
            return np.float64(0.0)
        idx = bisect_right(_ATMO_ALTS, h_km) - 1
        h0, rho0, H = _ATMO[idx]
        return np.float64(rho0 * exp(-(h_km - h0) / H))
