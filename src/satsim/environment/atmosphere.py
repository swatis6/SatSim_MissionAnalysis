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


# ---------------------------------------------------------------------------
# Species densities — Jacchia 1977 (port of j77sri.m by Huestis/SRI/Mahooti)
# ---------------------------------------------------------------------------

_AVOGADRO = 6.02214076e23   # molecules / mol

# Molecular weights [g/mol]
_WM0  = 28.96
_WMN2 = 28.0134
_WMO2 = 31.9988
_WMO  = 15.9994
_WMAR = 39.948
_WMHE = 4.0026
_WMH  = 1.0079

# Sea-level volume fractions
_QN2 = 0.78110
_QO2 = 0.20955
_QAR = 0.009343
_QHE = 0.000005242

_PI2 = math.pi / 2.0


@dataclass
class SpeciesDensities:
    """
    Number densities [cm⁻³] and derived quantities at one altitude.

    Attributes
    ----------
    altitude_km : float
    temperature_K : float
    N2, O2, O, Ar, He, H : float   number densities [cm⁻³]
    total : float                   total number density CM [cm⁻³]
    molecular_weight : float        mean molecular weight WM [g/mol]
    """
    altitude_km:      float
    temperature_K:    float
    N2:               float
    O2:               float
    O:                float
    Ar:               float
    He:               float
    H:                float
    total:            float
    molecular_weight: float

    @property
    def density_kg_m3(self) -> float:
        """Total mass density [kg/m³], suitable for drag calculations."""
        return self.total * self.molecular_weight * 1e3 / _AVOGADRO


def j77sri(Tinf: float, max_alt_km: int = 2500) -> list[SpeciesDensities]:
    """
    Jacchia 1977 atmosphere model (port of j77sri.m, SRI International).

    Given the exospheric temperature, returns number densities of N2, O2,
    O, Ar, He, H [cm⁻³] and temperature [K] at every integer km from 0 to
    max_alt_km. Use exospheric_temperature() to compute Tinf first.

    Below 86 km the US Standard Atmosphere 1976 is used; above 90 km the
    Jacchia 1977 diffusive-equilibrium model takes over, with empirical
    corrections to O2 and O. H atoms are included when max_alt_km >= 500.

    Parameters
    ----------
    Tinf : float
        Exospheric temperature T-infinity [K].
    max_alt_km : int
        Highest altitude to compute [km]. Default 2500.

    Returns
    -------
    list of SpeciesDensities, index i = altitude in km.
    """
    maxz = max_alt_km

    T   = [0.0] * (maxz + 1)
    CN2 = [0.0] * (maxz + 1)
    CO2 = [0.0] * (maxz + 1)
    CO  = [0.0] * (maxz + 1)
    CAr = [0.0] * (maxz + 1)
    CHe = [0.0] * (maxz + 1)
    CH  = [0.0] * (maxz + 1)
    CM  = [0.0] * (maxz + 1)
    WM  = [0.0] * (maxz + 1)

    E5M = [0.0] * 11   # polynomial mean molecular weight, iz = 90..100
    E6P = [0.0] * 11   # number density seed,             iz = 90..100

    for iz in range(maxz + 1):
        z   = float(iz)
        CH[iz] = 0.0

        # ----------------------------------------------------------------
        # 0–85 km: US Standard Atmosphere 1976
        # ----------------------------------------------------------------
        if iz <= 85:
            h = z * 6369.0 / (z + 6369.0)   # geopotential altitude [km]

            if iz <= 11:
                hbase, pbase, tbase, tgrad = 0.0,  1.0,          288.15, -6.5
            elif iz <= 20:
                hbase, pbase, tbase, tgrad = 11.0, 2.233611e-1,  216.65,  0.0
            elif iz <= 32:
                hbase, pbase, tbase, tgrad = 20.0, 5.403295e-2,  216.65,  1.0
            elif iz <= 47:
                hbase, pbase, tbase, tgrad = 32.0, 8.5666784e-3, 228.65,  2.8
            elif iz <= 51:
                hbase, pbase, tbase, tgrad = 47.0, 1.0945601e-3, 270.65,  0.0
            elif iz <= 71:
                hbase, pbase, tbase, tgrad = 51.0, 6.6063531e-4, 270.65, -2.8
            else:
                hbase, pbase, tbase, tgrad = 71.0, 3.9046834e-5, 214.65, -2.0

            if tgrad == 0.0:
                T[iz] = tbase
                x = math.exp(-34.163195 * (h - hbase) / tbase)
            else:
                T[iz] = tbase + tgrad * (h - hbase)
                x = (tbase / T[iz]) ** (34.163195 / tgrad)

            CM[iz] = 2.547e19 * (288.15 / T[iz]) * pbase * x
            y  = 10.0 ** (-3.7469 + (iz - 85) * (0.226434 - (iz - 85) * 5.945e-3))
            xf = 1.0 - y
            WM[iz]  = _WM0 * xf
            CN2[iz] = _QN2 * CM[iz]
            CO[iz]  = 2.0 * y * CM[iz]
            CO2[iz] = (xf * _QO2 - y) * CM[iz]
            CAr[iz] = _QAR * CM[iz]
            CHe[iz] = _QHE * CM[iz]

        # ----------------------------------------------------------------
        # 86–89 km: transition zone (oxygen dissociation ramp)
        # ----------------------------------------------------------------
        elif iz <= 89:
            T[iz] = 188.0
            y = 10.0 ** (-3.7469 + (iz - 85) * (0.226434 - (iz - 85) * 5.945e-3))
            WM[iz] = _WM0 * (1.0 - y)
            CM[iz] = CM[iz-1] * (T[iz-1] / T[iz]) * (WM[iz] / WM[iz-1]) * math.exp(
                -0.5897446 * (
                    (WM[iz-1] / T[iz-1]) * (1.0 + (iz - 1) / 6356.766) ** (-2)
                    + (WM[iz]  / T[iz])  * (1.0 +  iz       / 6356.766) ** (-2)
                )
            )
            xf      = 1.0 - y
            WM[iz]  = _WM0 * xf
            CN2[iz] = _QN2 * CM[iz]
            CO[iz]  = 2.0 * y * CM[iz]
            CO2[iz] = (xf * _QO2 - y) * CM[iz]
            CAr[iz] = _QAR * CM[iz]
            CHe[iz] = _QHE * CM[iz]

        # ----------------------------------------------------------------
        # 90+ km: Jacchia 1977 diffusive equilibrium
        # ----------------------------------------------------------------
        else:
            # Temperature profile
            if iz == 90 or Tinf < 188.1:
                T[iz] = 188.0
            else:
                xv  = 0.0045 * (Tinf - 188.0)
                Tx  = 188.0 + 110.5 * math.log(xv + math.sqrt(xv * xv + 1.0))
                Gx  = _PI2 * 1.9 * (Tx - 188.0) / (125.0 - 90.0)
                if iz <= 125:
                    T[iz] = Tx + ((Tx - 188.0) / _PI2) * math.atan(
                        (Gx / (Tx - 188.0)) * (iz - 125.0)
                        * (1.0 + 1.7 * ((iz - 125.0) / (iz - 90.0)) ** 2)
                    )
                else:
                    T[iz] = Tx + ((Tinf - Tx) / _PI2) * math.atan(
                        (Gx / (Tinf - Tx)) * (iz - 125.0)
                        * (1.0 + 5.5e-5 * (iz - 125.0) ** 2)
                    )

            # Number densities
            if iz <= 100:
                idx = iz - 90
                E5M[idx] = 28.89122 + idx * (
                    -2.83071e-2 + idx * (
                        -6.59924e-3 + idx * (
                            -3.39574e-4 + idx * (6.19256e-5 + idx * (-1.84796e-6))
                        )
                    )
                )
                if iz == 90:
                    E6P[0] = 7.145e13 * T[90]
                else:
                    G0 = (1.0 + (iz - 1) / 6356.766) ** (-2)
                    G1 = (1.0 +  iz      / 6356.766) ** (-2)
                    E6P[idx] = E6P[idx - 1] * math.exp(
                        -0.5897446 * (G1 * E5M[idx] / T[iz] + G0 * E5M[idx - 1] / T[iz - 1])
                    )
                x       = E5M[idx] / _WM0
                y       = E6P[idx] / T[iz]
                CN2[iz] = _QN2 * y * x
                CO[iz]  = 2.0 * (1.0 - x) * y
                CO2[iz] = (x * (1.0 + _QO2) - 1.0) * y
                CAr[iz] = _QAR * y * x
                CHe[iz] = _QHE * y * x
            else:
                G0 = (1.0 + (iz - 1) / 6356.766) ** (-2)
                G1 = (1.0 +  iz      / 6356.766) ** (-2)
                x       = 0.5897446 * (G1 / T[iz] + G0 / T[iz - 1])
                y       = T[iz - 1] / T[iz]
                CN2[iz] = CN2[iz - 1] * y * math.exp(-_WMN2 * x)
                CO2[iz] = CO2[iz - 1] * y * math.exp(-_WMO2 * x)
                CO[iz]  = CO[iz - 1]  * y * math.exp(-_WMO  * x)
                CAr[iz] = CAr[iz - 1] * y * math.exp(-_WMAR * x)
                CHe[iz] = CHe[iz - 1] * (y ** 0.62) * math.exp(-_WMHE * x)

    # ----------------------------------------------------------------
    # Empirical corrections to O2 and O (Jacchia 1977)
    # ----------------------------------------------------------------
    for iz in range(90, maxz + 1):
        CO2[iz] *= 10.0 ** (-0.07 * (1.0 + math.tanh(0.18 * (iz - 111.0))))
        CO[iz]  *= 10.0 ** (-0.24 * math.exp(-0.009 * (iz - 97.7) ** 2))
        CM[iz]   = CN2[iz] + CO2[iz] + CO[iz] + CAr[iz] + CHe[iz] + CH[iz]
        if CM[iz] > 0.0:
            WM[iz] = (
                _WMN2 * CN2[iz] + _WMO2 * CO2[iz] + _WMO  * CO[iz]
                + _WMAR * CAr[iz] + _WMHE * CHe[iz] + _WMH * CH[iz]
            ) / CM[iz]

    # ----------------------------------------------------------------
    # H atom densities (Jacchia 1977, only when maxz >= 500)
    # ----------------------------------------------------------------
    if maxz >= 500:
        phid00 = 10.0 ** (6.9 + 28.9 * Tinf ** (-0.25)) / 2e20 * 5.24e2
        H_500  = 10.0 ** (-0.06 + 28.9 * Tinf ** (-0.25))

        for iz in range(150, maxz + 1):
            phid0  = phid00 / math.sqrt(T[iz])
            WM[iz] = _WMH * 0.5897446 * (1.0 + iz / 6356.766) ** (-2) / T[iz] + phid0
            CM[iz] = CM[iz] * phid0

        y = WM[150]; WM[150] = 0.0
        for iz in range(151, maxz + 1):
            x = WM[iz - 1] + (y + WM[iz])
            y = WM[iz]
            WM[iz] = x

        for iz in range(150, maxz + 1):
            WM[iz] = math.exp(WM[iz]) * (T[iz] / T[150]) ** 0.75
            CM[iz] = WM[iz] * CM[iz]

        y = CM[150]; CM[150] = 0.0
        for iz in range(151, maxz + 1):
            x = CM[iz - 1] + 0.5 * (y + CM[iz])
            y = CM[iz]
            CM[iz] = x

        for iz in range(150, maxz + 1):
            CH[iz] = (WM[500] / WM[iz]) * (H_500 - (CM[iz] - CM[500]))

        for iz in range(150, maxz + 1):
            CM[iz] = CN2[iz] + CO2[iz] + CO[iz] + CAr[iz] + CHe[iz] + CH[iz]
            if CM[iz] > 0.0:
                WM[iz] = (
                    _WMN2 * CN2[iz] + _WMO2 * CO2[iz] + _WMO  * CO[iz]
                    + _WMAR * CAr[iz] + _WMHE * CHe[iz] + _WMH * CH[iz]
                ) / CM[iz]

    return [
        SpeciesDensities(
            altitude_km=float(iz),
            temperature_K=T[iz],
            N2=CN2[iz], O2=CO2[iz], O=CO[iz],
            Ar=CAr[iz], He=CHe[iz], H=CH[iz],
            total=CM[iz],
            molecular_weight=WM[iz],
        )
        for iz in range(maxz + 1)
    ]


def species_at_altitude(Tinf: float, altitude_km: float) -> SpeciesDensities:
    """
    Return species densities at a single altitude.

    Computes the full Jacchia 1977 profile from 0 km up to altitude_km,
    then returns the entry at the requested altitude.

    Parameters
    ----------
    Tinf : float
        Exospheric temperature [K] from exospheric_temperature().
    altitude_km : float
        Target altitude [km].
    """
    alt = max(0, int(altitude_km))
    return j77sri(Tinf, max_alt_km=alt)[alt]
