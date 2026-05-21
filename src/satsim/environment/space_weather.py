from __future__ import annotations

import io
import os
import requests
import pandas as pd
from atmosphere import SolarFlux, GeomagneticActivity

_GFZ_URL = 'https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt'

_LOCAL_CACHE = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', 'data',
    'Kp_ap_Ap_SN_F107_since_1932.txt',
))

_GFZ_COLS = [
    'YYYY', 'MM', 'DD', 'days', 'days_m', 'Bsr', 'dB',
    'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8',
    'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7', 'ap8',
    'Ap', 'SN', 'F10.7obs', 'F10.7adj', 'D',
]

_DEFAULT_FLUX = SolarFlux(f107_daily=70.0, f107_81day_avg=70.0)
_DEFAULT_GEO  = GeomagneticActivity(ap_current=4.0, ap_history=(4.0,) * 8)

_SW_DICT: dict | None = None


def load_space_weather_csv(source: str) -> dict:
    """
    Load a GFZ Kp/Ap/F10.7 file from a local path or URL.

    Computes the 81-day centred F10.7 average since GFZ does not include it.
    """
    df = pd.read_csv(
        source,
        comment='#',
        sep=r'\s+',
        header=None,
        names=_GFZ_COLS,
    )

    df = df[df['F10.7adj'] > 0].copy()

    df['DATE'] = (
        df['YYYY'].astype(str) + '-'
        + df['MM'].astype(str).str.zfill(2) + '-'
        + df['DD'].astype(str).str.zfill(2)
    )

    df['F10.7_81day_avg'] = (
        df['F10.7adj']
        .rolling(window=81, center=True, min_periods=41)
        .mean()
        .fillna(df['F10.7adj'])
    )

    return df.set_index('DATE').to_dict('index')


def get_sw_dict() -> dict:
    """
    Return the space weather dict, loading on first call.

    Tries the local cache first; if not found fetches directly from GFZ online.
    The result is cached in memory for the rest of the session.
    """
    global _SW_DICT
    if _SW_DICT is not None:
        return _SW_DICT

    if os.path.exists(_LOCAL_CACHE):
        print(f"Loading space weather from local cache: {_LOCAL_CACHE}")
        _SW_DICT = load_space_weather_csv(_LOCAL_CACHE)
    else:
        print(f"Local cache not found. Fetching from {_GFZ_URL} ...")
        response = requests.get(_GFZ_URL, verify=False, timeout=60)
        response.raise_for_status()
        _SW_DICT = load_space_weather_csv(io.StringIO(response.text))
        print("Done.")

    return _SW_DICT


def get_jacchia_drivers(
    date_str: str,
    sw_dict: dict,
) -> tuple[SolarFlux, GeomagneticActivity]:
    """
    Return SolarFlux and GeomagneticActivity for a given date, ready to pass
    into exospheric_temperature(). Falls back to solar-minimum defaults if the
    date is not in the dataset (e.g. future simulations).

    Parameters
    ----------
    date_str : str
        Date in 'YYYY-MM-DD' format.
    sw_dict : dict
        Output of get_sw_dict() or load_space_weather_csv().
    """
    row = sw_dict.get(date_str)
    if row is None:
        print(f"Warning: {date_str} not in space weather data, using solar-minimum defaults.")
        return _DEFAULT_FLUX, _DEFAULT_GEO

    return (
        SolarFlux(
            f107_daily=float(row['F10.7adj']),
            f107_81day_avg=float(row['F10.7_81day_avg']),
        ),
        GeomagneticActivity(
            ap_current=float(row['ap1']),
            ap_history=tuple(float(row[f'ap{i}']) for i in range(1, 9)),
        ),
    )
