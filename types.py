import astropy.time
import astropy.units
import numpy as np

from satsim.types import Vec3

# https://www.iau.org/static/resolutions/IAU2012_English.pdf
AU_KM = np.float64(149_597_870.7)  # 1 AU in km
_J2000_JD = np.float64(2_451_545.0)


# https://aa.usno.navy.mil/faq/sun_approx
# ECI -- Earth Centered Inertial
#  - Origin: Earth's center of mass
#  - X-axis: points toward the vernal equinox (where the sun crosses the equatorial plane at spring)
#  - Z-axis: points along Earth's north pole (rotation axis)
#  - Y-axis: completes the right-hand system
# calculates approximate solar coordinates
# sun position ECI at epoch + t seconds.
def sun_position_eci(epoch: astropy.time.Time, t: float) -> Vec3:
    current = epoch + t * astropy.units.s
    days_from_j2000 = current.jd - _J2000_JD

    mean_anomaly = np.radians(357.529 + 0.98560028 * days_from_j2000)
    mean_longitude_deg = 280.459 + 0.98564736 * days_from_j2000
    ecliptic_longitude = np.radians(
        mean_longitude_deg
        + 1.915 * np.sin(mean_anomaly)
        + 0.020 * np.sin(2 * mean_anomaly)
    )
    distance_au = (
        1.00014 - 0.01671 * np.cos(mean_anomaly) - 0.00014 * np.cos(2 * mean_anomaly)
    )
    obliquity = np.radians(23.439 - 0.00000036 * days_from_j2000)

    distance_km = distance_au * AU_KM
    return np.array(
        [
            distance_km * np.cos(ecliptic_longitude),
            distance_km * np.cos(obliquity) * np.sin(ecliptic_longitude),
            distance_km * np.sin(obliquity) * np.sin(ecliptic_longitude),
        ]
    )


def in_shadow(sat_pos: Vec3, sun_pos: Vec3, earth_radius: float) -> bool:
    """Cylindrical Earth shadow test. True if satellite is in Earth's shadow."""
    sun_direction = sun_pos / np.linalg.norm(sun_pos)
    projection = np.dot(sat_pos, sun_direction)
    if projection >= 0:
        return False  # satellite is on the sun-side of Earth
    perpendicular_dist = np.linalg.norm(sat_pos - projection * sun_direction)
    return bool(perpendicular_dist < earth_radius)
