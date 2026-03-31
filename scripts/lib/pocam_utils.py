"""
pocam_utils.py
==============
Shared utilities for POCAM analysis.
Includes: sigmoid model, correction curve fitting, datetime conversion,
          data loading, and photon number helpers.
"""

import numpy as np
import scipy.integrate
from scipy import special
from datetime import datetime
from scipy.optimize import curve_fit
import pytz
import h5py
import math

# ---------------------------------------------------------------------------
# HDF5 key helpers  (mirrors tdc.py)
# ---------------------------------------------------------------------------

def _lmg_key(driver, pwm, coarse, fine, temp):
    return f'{driver}/{pwm}/{coarse}-{fine}/{temp}C'


def _kapu_key(driver, pwm, mode, temp):
    return f'{driver}/{pwm}/{mode}/{temp}C'


def _build_key(diode, driver, pwm, coarse, fine, mode, temp):
    if 'LMG' in diode:
        return _lmg_key(driver, pwm, coarse, fine, temp)
    return _kapu_key(driver, pwm, mode, temp)


def _resolve_target(h5file, diode, pwm, coarse, fine, mode, temp):
    """Try target '1', fall back to '2'. Return (target_str, driver_str)."""
    prefix = 'lmg' if 'LMG' in diode else 'kapu'
    for t in ('1', '2'):
        driver = prefix + t
        key    = _build_key(diode, driver, pwm, coarse, fine, mode, temp)
        try:
            _ = h5file[key]
            return t, driver
        except KeyError:
            continue
    raise KeyError(f'No data found for {diode} at {temp}°C '
                   f'(tried both targets) in {h5file.filename}')


# ---------------------------------------------------------------------------
# Correction data (air/ice transmission curves) — loaded from files.
# ---------------------------------------------------------------------------

X_PRE, Y_PRE       = np.loadtxt('files/data_pre.txt',    unpack=True)
X_VALUES, Y_VALUES  = np.loadtxt('files/data_values.txt', unpack=True)


# ---------------------------------------------------------------------------
# Correction curves
# ---------------------------------------------------------------------------

def fit_correction_curves(x_pre=X_PRE, y_pre=Y_PRE,
                          x_values=X_VALUES, y_values=Y_VALUES):
    """
    Fit sigmoid curves to air and ice correction data.

    Returns
    -------
    popt_air, popt_ice : tuple
        Optimal sigmoid parameters for air and ice respectively.
    """
    x_air = np.concatenate([np.flip(x_pre), x_values[::2]])[10:]
    x_ice = np.concatenate([np.flip(x_pre), x_values[1::2]])
    y_air = np.concatenate([np.flip(y_pre), y_values[::2]])[10:]
    y_ice = np.concatenate([np.flip(y_pre), y_values[1::2]])

    p0 = [1.1, -1.0, 95, -0.001]
    popt_air, _ = curve_fit(sigmoid, x_air, y_air, p0=p0, maxfev=1000)
    popt_ice, _ = curve_fit(sigmoid, x_ice, y_ice, p0=p0, maxfev=1000)
    return popt_air, popt_ice

def apply_ice_correction(data, angles, popt_ice, popt_air):
    """
    Apply ice-to-air correction factor element-wise.

    Parameters
    ----------
    data             : np.ndarray  raw normalised data
    angles           : np.ndarray  zenith angles (degrees) matching data shape
    popt_ice, popt_air : sigmoid parameters from fit_correction_curves()

    Returns
    -------
    np.ndarray — corrected data
    """
    return data * sigmoid(angles, *popt_ice) / sigmoid(angles, *popt_air)


# ---------------------------------------------------------------------------
# Datetime conversion
# ---------------------------------------------------------------------------

def date_to_unix(date_str, tz_name='Europe/Berlin'):
    """
    Convert 'YYYY-MM-DD_HH_MM_SS' (local time) to a UTC Unix timestamp.
    Used by isotropy classes (cali_flange date format).
    """
    dt_naive = datetime.strptime(date_str, '%Y-%m-%d_%H_%M_%S')
    local_tz = pytz.timezone(tz_name)
    dt_local = local_tz.localize(dt_naive)
    return dt_local.astimezone(pytz.utc).timestamp()


def datetime_to_unix(dt_str, tz_name='Europe/Berlin'):
    """
    Convert 'YYYY-MM-DD HH:MM:SS.ffffff' (local time) to a UTC Unix timestamp.
    Used by photon classes (PD/PMT metadata datetime format).
    """
    dt_naive = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
    local_tz = pytz.timezone(tz_name)
    dt_local = local_tz.localize(dt_naive)
    return dt_local.astimezone(pytz.utc).timestamp()


def format_date(dt_str):
    """Return compact 'YYYYMMDD' from a 'YYYY-MM-DD ...' datetime string."""
    return dt_str.split(' ')[0].replace('-', '')


# ---------------------------------------------------------------------------
# Isotropy computation
# ---------------------------------------------------------------------------

def compute_isotropy(y_total, y_total_err):
    """
    Compute the isotropy value and its propagated uncertainty.

    The isotropy is defined as half the peak-to-peak range of the
    normalised angular emission profile.

    Parameters
    ----------
    y_total     : np.ndarray — combined (summed) emission data
    y_total_err : np.ndarray — combined uncertainty

    Returns
    -------
    y_norm       : np.ndarray
    y_norm_err   : np.ndarray
    isotropy     : float
    isotropy_err : float
    """
    mean   = np.mean(y_total)
    N      = y_total.size
    y_norm = y_total / mean

    total_var  = np.sum(y_total_err ** 2)
    y_norm_err = np.sqrt(
        ((mean - y_total / N) / mean ** 2) ** 2 * y_total_err ** 2
        + (y_total / (N * mean ** 2)) ** 2 * (total_var - y_total_err ** 2)
    )

    i_max = np.unravel_index(np.argmax(y_norm), y_norm.shape)
    i_min = np.unravel_index(np.argmin(y_norm), y_norm.shape)

    isotropy     = (y_norm[i_max] - y_norm[i_min]) / 2
    isotropy_err = np.sqrt(y_norm_err[i_max] ** 2 + y_norm_err[i_min] ** 2) / 2

    return y_norm, y_norm_err, float(isotropy), float(isotropy_err)


# ---------------------------------------------------------------------------
# Photon number helpers
# ---------------------------------------------------------------------------

# NIST photodiode responsivity  {emitter: [R [A/W], delta_R [A/W]]}
NIST = {
    'LMG365':  [0.1416, 0.00380],
    'LMG405':  [0.1800, 0.00300],
    'LMG450':  [0.2111, 0.00240],
    'LMG520':  [0.2610, 0.00100],
    'KAPU405': [0.1800, 0.00300],
    'KAPU465': [0.2220, 0.00200],
}

HC = 1.98644586e-25   # h·c  [m³·kg/s²]

# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def sigmoid(x, k, w, x0, y0):
    """Sigmoid function used for air/ice correction curve fitting."""
    return k / (1.0 + np.exp(-w * (x - x0))) + y0

def int_func(x, k, w, x0, y0):
    """Sigmoid × sin(x) — integrand for solid-angle integration."""
    return sigmoid(x, k, w, x0, y0) * np.sin(x)

def integrate_sigmoid(popt):
    """Integrate _int_func from 0 to π."""
    return scipy.integrate.quad(lambda x: int_func(x, *popt), 0, np.pi)[0]


def calc_photons(current, wavelength_m, responsivity, pulse_time, geo_factor):
    """
    Compute the number of photons hitting the PD.

    Parameters
    ----------
    current       : float  PD current [A]
    wavelength_m  : float  Wavelength [m]
    responsivity  : float  NIST responsivity [A/W]
    pulse_time    : float  Pulse duration [s]
    geo_factor    : float  d² / A_PD

    Returns
    -------
    float — photon count
    """
    return current * pulse_time * wavelength_m * geo_factor / (responsivity * HC)


def integrate_solid_angle(popt):
    """
    Integrate int_func over [0, π] (full upper hemisphere solid angle).

    Parameters
    ----------
    popt : array-like  sigmoid parameters [k, w, x0, y0]

    Returns
    -------
    float — integral value
    """
    return scipy.integrate.quad(lambda x: int_func(x, *popt), 0, np.pi)[0]


# ---------------------------------------------------------------------------
# HDF5 helper
# ---------------------------------------------------------------------------

def load_hdf5(path):
    """Open an HDF5 file in read mode. Caller is responsible for closing."""
    return h5py.File(path, 'r')


# ---------------------------------------------------------------------------
# Fit model functions
# ---------------------------------------------------------------------------

def gaussian(x, mean, sigma):
    return 1 / (np.sqrt(2 * math.pi) * sigma) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def skewing_erf(x, mean, sigma, alpha):
    return 0.5 * (1 + special.erf(alpha * (x - mean) / (np.sqrt(2) * sigma)))


def skewed_gaussian(x, mean, sigma, scaling, alpha):
    """Skewed Gaussian: scaled pdf of a skew-normal distribution."""
    return scaling * 2 * gaussian(x, mean, sigma) * skewing_erf(x, mean, sigma, alpha)


def skewed_double_gaussian(x, mean, sigma, scaling, alpha,
                           mean_2, sigma_2, scaling_2):
    """Skewed Gaussian main peak + plain Gaussian secondary peak."""
    return (skewed_gaussian(x, mean, sigma, scaling, alpha)
            + scaling_2 * gaussian(x, mean_2, sigma_2))