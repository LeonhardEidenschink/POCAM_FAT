"""
spectrometer_lib.py
===================
Library for POCAM spectrometer (spectral profile) calculations.
Contains the SingleSpecData class, fit helpers, and the JSON-writing routine.
Import this from your run script — do not run directly.
"""

import json
import math
import os

import h5py as h5
import numpy as np
import scipy.interpolate
from scipy.optimize import curve_fit

from pocam_utils import datetime_to_unix, format_date


# ---------------------------------------------------------------------------
# Spectrometer resolution [nm]  (Pyro-inferred values per emitter)
# ---------------------------------------------------------------------------

SPEC_RESOLUTION = {
    '365':     (4.11, 0.98),
    'LMG405':  (2.80, 0.77),
    'KAPU405': (2.80, 0.77),
    '450':     (2.48, 0.78),
    '465':     (2.66, 0.82),
    '520':     (3.31, 0.05),
}


def _resolution(diode):
    """Return (resolution, resolution_err) [nm] for the given emitter string."""
    for key, val in SPEC_RESOLUTION.items():
        if key in diode:
            return val
    raise ValueError(f"No spectrometer resolution entry found for diode '{diode}'")


# ---------------------------------------------------------------------------
# Wavelength-shift calibration model
# ---------------------------------------------------------------------------

# Polynomial coefficients for wl_shift = f(reconstructed_wl)
_SHIFT_COEFFS = (1.75676707e-10, -4.24495837e-07,  4.06196466e-04,
                 -1.92362595e-01,  4.50686924e+01, -4.17907302e+03)

# Polynomial coefficients for shift uncertainty
_ERR_COEFFS   = (-9.45984094e-11,  2.19810338e-07, -2.01225610e-04,
                  9.07280278e-02, -2.01617985e+01,  1.77025995e+03)


def wl_shift_model(x):
    """
    Map reconstructed CWL [nm] → (true_wl [nm], uncertainty [nm]).

    Reliable in the 370–530 nm range.
    """
    a, b, c, d, e, f = _SHIFT_COEFFS
    shift   = a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
    true_wl = x + shift

    z, y, w, v, u, t = _ERR_COEFFS
    err = abs(z*x**5 + y*x**4 + w*x**3 + v*x**2 + u*x + t)
    return float(true_wl), float(err)


# ---------------------------------------------------------------------------
# Gaussian fit models
# ---------------------------------------------------------------------------

def simple_gaussian(x, mean, sigma, scaling):
    """Gaussian without spectrometer resolution convolution."""
    return (scaling / (np.sqrt(2 * np.pi) * sigma)
            * np.exp(-0.5 * ((x - mean) / sigma) ** 2))


def gaussian_with_resolution(x, mean, sigma, scaling, resolution):
    """Gaussian with spectrometer resolution convolved in quadrature."""
    sigma_eff = np.sqrt(resolution**2 + sigma**2)
    return (scaling / (np.sqrt(2 * np.pi) * sigma_eff)
            * np.exp(-0.5 * ((x - mean) / sigma_eff) ** 2))


# ---------------------------------------------------------------------------
# HDF5 key helpers
# ---------------------------------------------------------------------------

def _lmg_key(driver, pwm, coarse, fine, temp):
    if temp == '25C_precheck':
        return f'{driver}/{pwm}/{coarse}-{fine}/{temp}'
    return f'{driver}/{pwm}/{coarse}-{fine}/{temp}C'


def _kapu_key(driver, pwm, mode, temp):
    if temp == '25C_precheck':
        return f'{driver}/{pwm}/{mode}/{temp}'
    return f'{driver}/{pwm}/{mode}/{temp}C'


# ---------------------------------------------------------------------------
# SingleSpecData
# ---------------------------------------------------------------------------

class SingleSpecData:
    """
    Load and process spectrometer data for one hemisphere/emitter/temperature.

    Parameters
    ----------
    hemisphere  : str    e.g. '39'
    pwm         : int    power setting
    temp        : int or str   temperature [°C] or '25C_precheck'
    diode       : str    e.g. 'LMG405'
    coarse      : int    LMG coarse setting
    fine        : int    LMG fine setting
    mode        : str    KAPU mode
    base_path   : str    path template with {hem} placeholder
    wavelength_path : str  path to wavelength_array.npy

    Attributes (selected)
    ---------------------
    average_signal_counts   : np.ndarray  normalised bg-subtracted spectrum
    wavelength              : np.ndarray  wavelength axis [nm]
    fit, cov                : Gaussian fit parameters and covariance
    fit_x_array, fit_y_array: fit curve for plotting
    cwl, cwl_err            : effective (reconstructed) CWL and error [nm]
    eff_fwhm, eff_fwhm_err  : effective FWHM and error [nm]
    pocam_cwl, pocam_cwl_err: true POCAM CWL after wavelength shift [nm]
    pocam_fwhm, pocam_fwhm_err: true POCAM FWHM after resolution deconvolution [nm]
    meas_time               : UTC Unix timestamp
    date                    : compact date string 'YYYYMMDD'
    target                  : '1' or '2'
    diode, pwm, coarse, fine, mode, temp : as passed
    """

    N_BINS = 288   # fixed spectrometer pixel count

    def __init__(self, hemisphere, pwm, temp, diode,batch,
                 coarse=1, fine=20, mode='default',
                 base_path=None, wavelength_path=None):

        self.diode       = diode
        self.nominal_cwl = int(diode[-3:])
        self.hemisphere  = hemisphere
        self.pwm         = pwm
        self.coarse      = coarse
        self.fine        = fine
        self.mode        = mode
        self.temp        = temp

        self.wavelength = np.load(wavelength_path)

        h5_path = base_path.format(batch=batch, hem=hemisphere)
        h = h5.File(h5_path + diode, 'r')

        # Try driver target '1', fall back to '2'
        for t in ('1', '2'):
            prefix = 'lmg' if 'LMG' in diode else 'kapu'
            driver = prefix + t
            try:
                if 'LMG' in diode:
                    key = _lmg_key(driver, pwm, coarse, fine, temp)
                else:
                    key = _kapu_key(driver, pwm, mode, temp)
                self.total_counts = np.array(h[key].get('spec_signal'))
                self.bg_counts    = np.array(h[key].get('spec_bg'))
                self.target       = t
                self.driver       = driver
                break
            except (KeyError, TypeError):
                continue

        # Measurement timestamp
        if 'LMG' in diode:
            meta_key = f'{self.driver}/{54000}/1-20/25C/metadata'
        else:
            meta_key = f'{self.driver}/{54000}/default/25C/metadata'
        dt_str = h[meta_key].attrs.get('datetime')
        h.close()

        self.date      = format_date(dt_str)
        self.meas_time = datetime_to_unix(dt_str)

        # ---- Background subtraction and normalisation ----
        # Handle both single-array and multi-cycle cases by ensuring 2-D
        bg = np.atleast_2d(self.bg_counts)
        sg = np.atleast_2d(self.total_counts)

        self.average_bg_counts = bg.mean(axis=0)
        signal                 = sg - self.average_bg_counts
        self.average_signal_counts = signal.mean(axis=0)

        norm = np.sum(self.average_signal_counts)
        self.norm = 1.0 / norm
        self.average_signal_counts *= self.norm
        self.bg_std = np.std(self.average_bg_counts) * self.norm
        self.y_err  = np.sqrt(np.abs(self.average_signal_counts))

        # ---- Peak and approximate width ----
        self.peak      = np.max(self.average_signal_counts)
        peak_idx       = np.argmax(self.average_signal_counts)
        self.peak_wl   = self.wavelength[peak_idx]

        interp = scipy.interpolate.InterpolatedUnivariateSpline(
            self.wavelength, self.average_signal_counts)

        wl_fine  = np.linspace(self.peak_wl - 30, self.peak_wl + 30, 100)
        y_fine   = interp(wl_fine)

        left  = wl_fine[wl_fine < self.peak_wl]
        right = wl_fine[wl_fine > self.peak_wl]
        self.start = left[np.argmin(np.abs(y_fine[:len(left)] - 0.5 * self.peak))]
        self.end   = right[np.argmin(np.abs(y_fine[-len(right):] - 0.5 * self.peak))]

        approx_fwhm = self.end - self.start
        approx_std  = approx_fwhm / 2.3548
        scaling_p0  = self.peak * np.sqrt(2 * math.pi) * approx_std ** 2

        # ---- Gaussian fit (narrow window around nominal CWL) ----
        cwl = self.nominal_cwl
        mask        = (self.wavelength > cwl - 20) & (self.wavelength < cwl + 20)
        fit_xdata   = self.wavelength[mask]
        fit_ydata   = self.average_signal_counts[mask]

        self.fit, self.cov = curve_fit(
            simple_gaussian,
            fit_xdata, fit_ydata,
            p0=[cwl, approx_std, scaling_p0],
            bounds=([cwl - 30, 0, 0], [cwl + 30, np.inf, np.inf]),
            maxfev=10000)

        self.fit_function = lambda x: simple_gaussian(x, *self.fit)
        self.fit_x_array  = np.linspace(cwl - 50, cwl + 50, 1001)
        self.fit_y_array  = self.fit_function(self.fit_x_array)

        # ---- Effective (measured) parameters ----
        self.cwl           = self.fit[0]
        self.cwl_err       = np.sqrt(abs(self.cov[0][0]))
        self.eff_std       = self.fit[1]
        self.eff_std_err   = np.sqrt(abs(self.cov[1][1]))
        self.eff_fwhm      = 2.3548 * self.eff_std
        self.eff_fwhm_err  = 2.3548 * self.eff_std_err

        # ---- True POCAM parameters ----
        res, res_err = _resolution(diode)
        self.resolution_spec     = res
        self.resolution_spec_err = res_err

        self.pocam_cwl, wl_err  = wl_shift_model(self.cwl)
        self.pocam_cwl_err      = np.sqrt(wl_err**2 + self.cwl_err**2)
        self.pocam_std          = np.sqrt(self.eff_std**2 - res**2)
        self.pocam_std_err      = np.sqrt(
            (self.eff_std / self.pocam_std)**2 * self.eff_std_err**2
            + (res / self.pocam_std)**2 * res_err**2)
        self.pocam_fwhm         = 2.3548 * self.pocam_std
        self.pocam_fwhm_err     = 2.3548 * self.pocam_std_err


# ---------------------------------------------------------------------------
# JSON assembly and file writing
# ---------------------------------------------------------------------------

def _driver_char(diode):
    return 'l' if 'LMG' in diode else 'k'


def compute_and_save(hemisphere, device_id, emitter,
                     pwm, temp, coarse, fine, mode,
                     paths, batch):
    """
    Compute spectral profile and write JSON output file.

    Parameters
    ----------
    hemisphere : str   HDF5 hemisphere id
    device_id  : str   POCAM device number e.g. '016'
    emitter    : str   e.g. 'LMG405'
    paths      : dict  with keys 'data', 'output', 'wavelength'

    Returns
    -------
    dict — assembled spectral distribution dictionary
    """
    base_path      = paths['data']
    wavelength_path = paths['wavelength']
    driver         = _driver_char(emitter)

    spec = SingleSpecData(
        hemisphere=hemisphere, pwm=pwm, temp=temp, diode=emitter,
        coarse=coarse, fine=fine, mode=mode,batch = batch,
        base_path=base_path, wavelength_path=wavelength_path)

    target_label = 'master' if spec.target == '1' else 'slave'

    # ---- graph-with-fit entry ----
    graph_entry = {
        'data_format':  'graph-with-fit',
        'power':         pwm,
        'temperature':   temp,
        'x_label':       'Wavelength [nm]',
        'y_label':       'Relative Counts',
        'x_values':      spec.wavelength.tolist(),
        'y_values':      spec.average_signal_counts.tolist(),
        'x_min':         float(np.min(spec.wavelength)),
        'x_max':         float(np.max(spec.wavelength)),
        'n_bins':        len(spec.wavelength),
        'fit_x_min':     float(np.min(spec.fit_x_array)),
        'fit_x_max':     float(np.max(spec.fit_x_array)),
        'fit_n_points':  len(spec.fit_x_array),
        'fit_y_values':  spec.fit_y_array.tolist(),
        'title':         'Spectral Profile',
    }
    if 'LMG' in emitter:
        graph_entry['coarse'] = coarse
        graph_entry['fine']   = fine
    else:
        graph_entry['mode'] = mode

    # ---- FWHM value entry ----
    fwhm_entry = {
        'data_format': 'value',
        'value':        round(spec.pocam_fwhm, 2),
        'error':        round(spec.pocam_fwhm_err, 2),
        'power':        pwm,
        'temperature':  temp,
        'label':        'FWHM',
        'title':        'Full-Width-Half-Maximum [nm]',
    }
    if 'LMG' in emitter:
        fwhm_entry['coarse'] = coarse
        fwhm_entry['fine']   = fine
    else:
        fwhm_entry['mode'] = mode

    # ---- CWL value entry ----
    cwl_entry = {
        'data_format': 'value',
        'value':        round(spec.pocam_cwl, 2),
        'error':        round(spec.pocam_cwl_err, 2),
        'power':        pwm,
        'temperature':  temp,
        'label':        'CWL',
        'title':        'Central-Wavelength [nm]',
    }
    if 'LMG' in emitter:
        cwl_entry['coarse'] = coarse
        cwl_entry['fine']   = fine
    else:
        cwl_entry['mode'] = mode

    result = {
        'device_uid':    f'pocam-{spec.date}_{device_id}',
        'subdevice_uid': f'pocam-led-{target_label}_{driver}-{emitter[-3:]}_{device_id}',
        'meas_name':     'led-spectral-profile',
        'meas_class':    'display',
        'meas_stage':    'calibration',
        'meas_group':    'spectral-profile',
        'meas_site':     'tum',
        'meas_time':     spec.meas_time,
        'meas_data':     [graph_entry, fwhm_entry, cwl_entry],
        'comments': [
            "Spectral profile of the light emission for the chosen L(E)D "
            "at the specified conditions.",
            "Experimental data: normalised (sum = 1), background-subtracted "
            "spectral counts at discrete wavelengths from the spectrometer.",
            "coarse/fine (LMG) or mode (KAPU): pulse-shape settings.",
            "power: PWM value (0–54000 ≈ 0–32 V).",
            "Fit function: simple Gaussian (mu, sigma, scaling). "
            "Not a normalised PDF.",
            "POCAM CWL and FWHM are corrected for spectrometer wavelength shift "
            "and resolution broadening respectively.",
            "Structure of device_uid: 'pocam-{date}_{device_number}'",
            "Structure of subdevice_uid: "
            "'pocam-led-{target}_{driver}-{wl}_{device}'",
        ],
        'support_files': [
            {
                'filetype': 'hdf5',
                'hostname': 'data.icecube.wisc.edu',
                'pathname': f'/data/exp/IceCubeUpgrade/commissioning/pocam/'
                            f'pocam_{device_id}/{target_label}_hemisphere/{emitter}',
                'comment':  'Raw HDF5 data. See POCAM documentation for details.',
            },
            {
                'filetype': 'pdf',
                'hostname': 'data.icecube.wisc.edu',
                'pathname': '/data/exp/IceCubeUpgrade/commissioning/pocam/'
                            'POCAM_documentation.pdf',
                'comment':  'POCAM documentation guide.',
            },
        ],
    }

    out_dir = os.path.join(paths['output'], f'{batch}',
                                f'pocam_{device_id}',
                                f'hem_{hemisphere}',
                                f'{emitter}')
    
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)  
    filename = f'spec_{temp}C.json'
    out_path = os.path.join(out_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f'  Saved → {out_path}')

    return result