from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from satsim.utilities.consts import R_EARTH


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_GAS_CONSTANT = 8.314        # universal gas constant [J/(mol·K)]
_MOLAR_MASS   = 0.029        # mean molar mass of upper-atmosphere air [kg/mol]
_GRAVITY      = 9.80665      # surface gravitational acceleration [m/s²]


# ---------------------------------------------------------------------------
# Layer 1 — Input data structures (Jacchia model)
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    Geographic position of a point in the atmosphere.

    All three coordinates are required by the Jacchia model:
    - altitude drives the exponential density profile
    - latitude enters the diurnal temperature correction via cos(latitude)
    - longitude combined with universal time gives local solar time,
      which sets the phase of the diurnal bulge

    Attributes
    ----------
    altitude_km : float
        Geodetic altitude above Earth's surface [km]. Valid range: 90–2500 km.
    latitude_deg : float
        Geographic latitude [deg]. Range: -90 (south pole) to +90 (north pole).
    longitude_deg : float
        Geographic longitude [deg]. Range: -180 to +180.
    """
    altitude_km:   float
    latitude_deg:  float
    longitude_deg: float


@dataclass
class Time:
    """
    Universal time of an atmospheric observation.

    All three fields are required by the Jacchia model:
    - year anchors the absolute date for index lookups and Julian date computation
    - day_of_year drives the semiannual variation
    - seconds_of_day is universal time (UT), which combined with longitude
      gives local solar time: LST_hours = (seconds_of_day / 3600) + longitude_deg / 15

    Attributes
    ----------
    year : int
        Calendar year (e.g. 2024).
    day_of_year : int
        Day within the year [1–365, or 366 for leap years].
    seconds_of_day : float
        Universal time expressed in seconds [0, 86400).
    """
    year:           int
    day_of_year:    int
    seconds_of_day: float


@dataclass
class SolarFlux:
    """
    Solar 10.7 cm radio flux indices for atmospheric heating.

    Two distinct values are required because the atmosphere responds to
    different timescales of solar activity:
    - f107_daily captures short-term fluctuations (flares, active regions)
    - f107_81day_avg captures the slow solar cycle trend

    Both enter Jacchia's exospheric temperature formula independently:
        T_exo = 383 + 3.32 * f107_81day_avg + 1.8 * (f107_daily - f107_81day_avg)

    The 81-day average sets the baseline; the daily deviation shifts it.
    Exospheric temperature then determines the scale height and density falloff.

    Attributes
    ----------
    f107_daily : float
        Current day's measured F10.7 flux [sfu]. Typical range: 70–300.
    f107_81day_avg : float
        81-day centered running average of F10.7 [sfu]. Typical range: 70–300.
    """
    f107_daily:     float
    f107_81day_avg: float


@dataclass
class GeomagneticActivity:
    """
    Geomagnetic Ap index for the current period and recent history.

    The Ap index is reported in 3-hour intervals. The atmosphere's response
    to a geomagnetic storm lags approximately 6 hours behind the storm itself,
    so a single current value is insufficient — recent history is required.

    Jacchia splits the geomagnetic heating into two terms:
    - an instantaneous term using ap_current
    - a lagged term using an exponentially-weighted sum over ap_history,
      where the value 6 hours ago (index 2) carries the most weight

    Attributes
    ----------
    ap_current : float
        3-hour Ap index for the current period [nT]. Range: 0–400.
    ap_history : tuple[float, ...]
        The 8 most recent 3-hour Ap values, ordered newest to oldest [nT].
        Covers 24 hours of history (8 × 3-hour periods).
    """
    ap_current: float
    ap_history: tuple[float, ...]


# ---------------------------------------------------------------------------
# Layer 2 — Solar geometry and time conversion
# ---------------------------------------------------------------------------

_DEG2RAD                 = math.pi / 180.0
_SOLAR_DECL_AMPLITUDE_RAD = 23.45 * _DEG2RAD   # maximum solar declination [rad]
_PHASE_LAG_RAD           = 30.0  * _DEG2RAD    # shifts diurnal peak from noon to ~2 pm
_ASYMMETRY_COEFF         = 0.02                 # slight morning/afternoon asymmetry
_DIURNAL_EXPONENT        = 2.2                  # Jacchia shape exponent n


@dataclass
class SolarGeometry:
    """
    Derived solar geometry for a given time and position.

    This is the output of Layer 2: raw inputs (year, day, seconds, lat, lon)
    are reduced to three clean quantities that the exospheric temperature
    formula in Layer 3 consumes directly.

    Attributes
    ----------
    declination_rad : float
        Solar declination [rad] — angle of the Sun above/below the equatorial plane.
    hour_angle_rad : float
        Solar hour angle [rad] — zero at local solar noon, positive in the morning.
    geometric_factor : float
        Dimensionless factor in [0, 1] expressing how close the point is to the
        diurnal heating maximum. Includes the phase lag and asymmetry corrections.
    """
    declination_rad:  float
    hour_angle_rad:   float
    geometric_factor: float


def solar_declination(time: Time) -> float:
    """Return solar declination [rad] for the given day of year."""
    return _SOLAR_DECL_AMPLITUDE_RAD * math.sin(
        2.0 * math.pi / 365.0 * (time.day_of_year - 81)
    )


def _hour_angle(time: Time, position: Position) -> float:
    """Return the solar hour angle [rad]. Zero at local solar noon."""
    lst_hours = (time.seconds_of_day / 3600.0) + position.longitude_deg / 15.0
    lst_hours = lst_hours % 24.0
    return (lst_hours - 12.0) * math.pi / 12.0


def diurnal_geometric_factor(
    position: Position,
    declination_rad: float,
    hour_angle_rad: float,
) -> float:
    """Return dimensionless geometric factor [0, 1] relative to the diurnal heating bulge."""
    lat_rad = position.latitude_deg * _DEG2RAD
    h_eff = (
        hour_angle_rad
        + _PHASE_LAG_RAD
        + _ASYMMETRY_COEFF * math.sin(hour_angle_rad + _PHASE_LAG_RAD)
    )
    cos_theta = (
        math.sin(lat_rad) * math.sin(declination_rad)
        + math.cos(lat_rad) * math.cos(declination_rad) * math.cos(h_eff)
    )
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    return math.cos(theta / 2.0) ** _DIURNAL_EXPONENT


def compute_solar_geometry(time: Time, position: Position) -> SolarGeometry:
    """Reduce raw time and position to the three geometric quantities Layer 3 needs."""
    decl = solar_declination(time)
    ha   = _hour_angle(time, position)
    gf   = diurnal_geometric_factor(position, decl, ha)
    return SolarGeometry(declination_rad=decl, hour_angle_rad=ha, geometric_factor=gf)


# ---------------------------------------------------------------------------
# Layer 3 — Exospheric temperature
# ---------------------------------------------------------------------------

_T_SOLAR_INTERCEPT   = 379.0   # K   — Jacchia baseline intercept
_T_SOLAR_MEAN_COEFF  = 3.24    # K/sfu — coefficient on 81-day mean F10.7
_T_SOLAR_DELTA_COEFF = 1.3     # K/sfu — coefficient on daily deviation from mean
_DIURNAL_AMPLITUDE   = 0.3     # fractional day/night temperature swing
_SA_SCALE            = 28.0    # K   — semiannual variation amplitude
_SA_ANNUAL_PHASE     = 0.09    # phase of the annual component [fraction of year]
_SA_SEMI_PHASE       = 0.53    # phase of the semiannual component [fraction of year]
_GEO_LINEAR_COEFF    = 1.0     # K/nT — linear geomagnetic heating coefficient
_GEO_SAT_AMPLITUDE   = 125.0   # K   — saturation ceiling of the exponential term
_GEO_SAT_DECAY       = 0.08    # 1/nT — decay constant of the saturation term
_GEO_LAG_HOURS       = 6.0     # hours — atmospheric response lag to geomagnetic storms
_AP_INTERVAL_HOURS   = 3.0     # hours — cadence of Ap reporting


def _solar_baseline(flux: SolarFlux) -> float:
    """Return quiet-time exospheric temperature [K] driven by solar flux."""
    return (
        _T_SOLAR_INTERCEPT
        + _T_SOLAR_MEAN_COEFF  * flux.f107_81day_avg
        + _T_SOLAR_DELTA_COEFF * (flux.f107_daily - flux.f107_81day_avg)
    )


def _diurnal_correction(t_solar: float, geometry: SolarGeometry) -> float:
    """Return diurnal temperature contribution [K]."""
    return t_solar * _DIURNAL_AMPLITUDE * geometry.geometric_factor


def _semiannual_correction(time: Time) -> float:
    """Return semiannual temperature correction [K]."""
    tau  = time.day_of_year / 365.0
    f_sa = (
        0.02
        * (1.0 + math.sin(2.0 * math.pi * (tau + _SA_ANNUAL_PHASE)))
        * (0.5 + math.sin(4.0 * math.pi * (tau + _SA_SEMI_PHASE)))
    )
    return _SA_SCALE * f_sa


def _lagged_ap(geo: GeomagneticActivity) -> float:
    """Return effective Ap index [nT] with 6-hour exponential lag."""
    decay   = _AP_INTERVAL_HOURS / _GEO_LAG_HOURS
    values  = (geo.ap_current,) + tuple(geo.ap_history)
    weights = [math.exp(-i * decay) for i in range(len(values))]
    norm    = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / norm


def _geomagnetic_correction(geo: GeomagneticActivity) -> float:
    """Return geomagnetic temperature correction [K] per Jacchia 1965."""
    ap_eff = _lagged_ap(geo)
    return (
        _GEO_LINEAR_COEFF * ap_eff
        + _GEO_SAT_AMPLITUDE * (1.0 - math.exp(-_GEO_SAT_DECAY * ap_eff))
    )


def exospheric_temperature(
    flux: SolarFlux,
    geo: GeomagneticActivity,
    time: Time,
    geometry: SolarGeometry,
) -> float:
    """
    Return the exospheric temperature T-infinity [K].

    This is the central quantity of the Jacchia model. All density
    calculations downstream are functions of this single value.
    """
    t_solar = _solar_baseline(flux)
    return (
        t_solar
        + _diurnal_correction(t_solar, geometry)
        + _semiannual_correction(time)
        + _geomagnetic_correction(geo)
    )


# ---------------------------------------------------------------------------
# AtmosphereModel — unified interface used by the simulation
# ---------------------------------------------------------------------------

class AtmosphereModel:
    """
    Atmospheric density model with two operating modes.

    Simple mode (default):
        Uses an exponential decay with a fixed scale height. Fast, requires
        no external data. Good enough for rough drag estimates.

    Jacchia mode (when jacchia_inputs is supplied to density()):
        Derives the scale height from the Jacchia 1965 exospheric temperature,
        which accounts for solar activity, time of day, season, and geomagnetic
        storms. More accurate, especially above 300 km.

    The scale height formula used in Jacchia mode:
        H = (R * T_exo) / (M * g)
    where R is the gas constant, T_exo is the Jacchia temperature, M is the
    mean molar mass of air, and g is surface gravity.
    """

    def __init__(self, config: dict):
        self.rho0         = config.get("rho0", 1.225)       # sea-level density [kg/m³]
        self.scale_height = config.get("scale_height", 8500.0)  # default scale height [m]

    def density(
        self,
        r: np.ndarray,
        jacchia_inputs: tuple[Position, Time, SolarFlux, GeomagneticActivity] | None = None,
    ) -> float:
        """
        Return atmospheric density [kg/m³] at position r.

        Parameters
        ----------
        r : np.ndarray
            ECI position vector [m].
        jacchia_inputs : tuple or None
            If provided, a (Position, Time, SolarFlux, GeomagneticActivity) tuple
            that activates the Jacchia model for a physics-accurate scale height.
            If None, falls back to the simple exponential model.
        """
        altitude = np.linalg.norm(r) - R_EARTH
        if altitude < 0:
            return self.rho0

        if jacchia_inputs is not None:
            position, time, flux, geo = jacchia_inputs
            geometry = compute_solar_geometry(time, position)
            T_exo    = exospheric_temperature(flux, geo, time, geometry)
            # Scale height grows with temperature: hotter atmosphere = puffier = more drag at altitude
            H = (_GAS_CONSTANT * T_exo) / (_MOLAR_MASS * _GRAVITY)
        else:
            H = self.scale_height

        return self.rho0 * np.exp(-altitude / H)
