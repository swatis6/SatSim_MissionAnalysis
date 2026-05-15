from __future__ import annotations

import math
from dataclasses import dataclass


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

    altitude_km: float
    latitude_deg: float
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

    year: int
    day_of_year: int
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

    f107_daily: float
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

_DEG2RAD = math.pi / 180.0
_SOLAR_DECL_AMPLITUDE_RAD = 23.45 * _DEG2RAD   # maximum solar declination [rad]
_PHASE_LAG_RAD = 30.0 * _DEG2RAD               # shifts diurnal peak from noon to ~2 pm
_ASYMMETRY_COEFF = 0.02                         # slight morning/afternoon asymmetry
_DIURNAL_EXPONENT = 2.2                         # Jacchia shape exponent n


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

    declination_rad: float
    hour_angle_rad: float
    geometric_factor: float


def solar_declination(time: Time) -> float:
    """
    Return solar declination [rad] for the given day of year.

    Uses a sinusoidal approximation centred on the vernal equinox (day 81).
    """
    return _SOLAR_DECL_AMPLITUDE_RAD * math.sin(
        2.0 * math.pi / 365.0 * (time.day_of_year - 81)
    )


def _hour_angle(time: Time, position: Position) -> float:
    """
    Return the solar hour angle [rad].

    Zero at local solar noon, positive in the morning, negative in the afternoon.
    Local solar time is computed as UT + longitude / 15.
    """
    lst_hours = (time.seconds_of_day / 3600.0) + position.longitude_deg / 15.0
    lst_hours = lst_hours % 24.0
    return (lst_hours - 12.0) * math.pi / 12.0


def diurnal_geometric_factor(
    position: Position,
    declination_rad: float,
    hour_angle_rad: float,
) -> float:
    """
    Return the dimensionless geometric factor [0, 1] positioning a point
    relative to the diurnal heating bulge.

    Applies the phase lag (shifting the peak to ~2 pm local solar time) and
    a small asymmetry correction, then evaluates Jacchia's cos^n(theta/2)
    shape function.
    """
    lat_rad = position.latitude_deg * _DEG2RAD

    # Phase lag + morning/afternoon asymmetry
    h_eff = (
        hour_angle_rad
        + _PHASE_LAG_RAD
        + _ASYMMETRY_COEFF * math.sin(hour_angle_rad + _PHASE_LAG_RAD)
    )

    # Angle from the sub-solar point via the spherical law of cosines
    cos_theta = (
        math.sin(lat_rad) * math.sin(declination_rad)
        + math.cos(lat_rad) * math.cos(declination_rad) * math.cos(h_eff)
    )
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)

    return math.cos(theta / 2.0) ** _DIURNAL_EXPONENT


def compute_solar_geometry(time: Time, position: Position) -> SolarGeometry:
    """
    Reduce raw time and position inputs to the three geometric quantities
    that the exospheric temperature formula (Layer 3) requires.
    """
    decl = solar_declination(time)
    ha = _hour_angle(time, position)
    gf = diurnal_geometric_factor(position, decl, ha)
    return SolarGeometry(
        declination_rad=decl,
        hour_angle_rad=ha,
        geometric_factor=gf,
    )


# ---------------------------------------------------------------------------
# Layer 3 — Exospheric temperature
# ---------------------------------------------------------------------------

_T_SOLAR_INTERCEPT   = 379.0   # K — Jacchia baseline intercept
_T_SOLAR_MEAN_COEFF  = 3.24    # K/sfu — coefficient on 81-day mean F10.7
_T_SOLAR_DELTA_COEFF = 1.3     # K/sfu — coefficient on daily deviation from mean

_DIURNAL_AMPLITUDE   = 0.3     # fractional day/night temperature swing

_SA_SCALE            = 28.0    # K — semiannual variation amplitude
_SA_ANNUAL_PHASE     = 0.09    # phase of the annual component [fraction of year]
_SA_SEMI_PHASE       = 0.53    # phase of the semiannual component [fraction of year]

_GEO_LINEAR_COEFF    = 1.0     # K/nT — linear geomagnetic heating coefficient (Jacchia 1965)
_GEO_SAT_AMPLITUDE   = 125.0   # K   — saturation ceiling of the exponential term
_GEO_SAT_DECAY       = 0.08    # 1/nT — decay constant of the saturation term
_GEO_LAG_HOURS       = 6.0     # hours — atmospheric response lag to geomagnetic storms
_AP_INTERVAL_HOURS   = 3.0     # hours — cadence of Ap reporting


def _solar_baseline(flux: SolarFlux) -> float:
    """
    Return the quiet-time exospheric temperature [K] driven by solar flux.

    The 81-day average sets the long-term baseline; the daily deviation
    adjusts it for short-term solar variability.
    """
    return (
        _T_SOLAR_INTERCEPT
        + _T_SOLAR_MEAN_COEFF  * flux.f107_81day_avg
        + _T_SOLAR_DELTA_COEFF * (flux.f107_daily - flux.f107_81day_avg)
    )


def _diurnal_correction(t_solar: float, geometry: SolarGeometry) -> float:
    """
    Return the diurnal temperature contribution [K].

    Scales the solar baseline by the geometric factor from Layer 2.
    The factor 0.3 gives a 30% swing between the night-side minimum and
    the sub-solar maximum.
    """
    return t_solar * _DIURNAL_AMPLITUDE * geometry.geometric_factor


def _semiannual_correction(time: Time) -> float:
    """
    Return the semiannual temperature correction [K].

    Combines an annual and a semiannual sinusoid to reproduce Jacchia's
    observed pattern of two density peaks per year (near the equinoxes).
    """
    tau = time.day_of_year / 365.0
    f_sa = (
        0.02
        * (1.0 + math.sin(2.0 * math.pi * (tau + _SA_ANNUAL_PHASE)))
        * (0.5 + math.sin(4.0 * math.pi * (tau + _SA_SEMI_PHASE)))
    )
    return _SA_SCALE * f_sa


def _lagged_ap(geo: GeomagneticActivity) -> float:
    """
    Return the effective representative Ap index [nT].

    Combines the current value with the 8-period history using an
    exponentially decaying weighted average (weights normalised to sum to 1).
    Recent values count more than older ones; the 6-hour-old value carries
    weight exp(-1) ≈ 0.37 relative to the current value.

    Jacchia 1965 notes the ~6-hour lag qualitatively but does not specify
    the exact averaging formula — that comes in MSIS-86 (Hedin 1987).
    """
    decay = _AP_INTERVAL_HOURS / _GEO_LAG_HOURS   # 0.5 per 3-hour interval
    values  = (geo.ap_current,) + tuple(geo.ap_history)
    weights = [math.exp(-i * decay) for i in range(len(values))]
    norm    = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / norm


def _geomagnetic_correction(geo: GeomagneticActivity) -> float:
    """
    Return the geomagnetic temperature correction [K] per Jacchia 1965.

    ΔT = 1.0 × ap_eff + 125 × [1 − exp(−0.08 × ap_eff)]

    The linear term gives proportional heating for moderate activity.
    The saturation term approaches 125 K asymptotically, preventing
    unrealistic heating during extreme storms.
    """
    ap_eff = _lagged_ap(geo)
    return (
        _GEO_LINEAR_COEFF  * ap_eff
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

    This is the central quantity of the Jacchia model. All density calculations
    downstream are functions of this single value.

    Parameters
    ----------
    flux : SolarFlux
        Daily and 81-day average F10.7 indices.
    geo : GeomagneticActivity
        Current and recent-history Ap indices.
    time : Time
        Year, day of year, and seconds of day (UT).
    geometry : SolarGeometry
        Pre-computed solar geometry from Layer 2.

    Returns
    -------
    float
        Exospheric temperature T-infinity [K].
    """
    t_solar = _solar_baseline(flux)
    return (
        t_solar
        + _diurnal_correction(t_solar, geometry)
        + _semiannual_correction(time)
        + _geomagnetic_correction(geo)
    )
