"""
tdc_lib.py
==========
Library for POCAM TDC (time-profile) calculations.
Contains the SingleTDCData class, fitting helpers, and the JSON-writing routine.
Import this from your run script — do not run directly.
"""

import json
import math
import os

import h5py as h5
import numpy as np
import scipy.interpolate
from numpy.random import multivariate_normal
from scipy import special
from scipy.optimize import curve_fit

from pocam_utils import datetime_to_unix, format_date, skewed_gaussian, skewed_double_gaussian


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

DELTA_BROADENING = 0.08   # TDC + APD broadening [ns]


# ---------------------------------------------------------------------------
# FWHM helpers
# ---------------------------------------------------------------------------

def compute_fwhm(fit_func, args, x_range):
    """Return FWHM of fit_func evaluated on x_range, or NaN on failure."""
    y = fit_func(x_range, *args)
    half_max = np.max(y) / 2
    idx = np.where(y >= half_max)[0]
    if len(idx) < 2:
        return np.nan
    return x_range[idx[-1]] - x_range[idx[0]]


def compute_fwhm_mc(fit_func, popt, pcov, x_range, n=1000):
    """
    Monte-Carlo uncertainty on FWHM by sampling [mean, sigma, alpha].

    Returns
    -------
    fwhm_mean : float
    fwhm_std  : float
    """
    mask = [0, 1, 3]   # mean, sigma, alpha
    popt_r = np.array(popt)[mask]
    pcov_r = pcov[np.ix_(mask, mask)]
    samples = multivariate_normal(popt_r, pcov_r, size=n)

    fwhms = []
    for s in samples:
        p = list(popt)
        p[0], p[1], p[3] = s[0], s[1], s[2]
        fwhms.append(compute_fwhm(fit_func, p, x_range))

    return float(np.mean(fwhms)), float(np.std(fwhms))


# ---------------------------------------------------------------------------
# Poissonian correction
# ---------------------------------------------------------------------------

def poissonian_correction(mean_occ, hist_counts, count_norm=9000):
    """Apply Poissonian dead-time correction to histogram counts."""
    corrected = []
    factors   = []
    for i in range(len(hist_counts)):
        factor = 1.0
        for j in range(i):
            factor *= np.exp(mean_occ / count_norm * corrected[j])
        corrected.append(factor * hist_counts[i])
        factors.append(factor)
    return np.array(corrected), np.array(factors)


# ---------------------------------------------------------------------------
# HDF5 key helpers  (shared with photons_lib)
# ---------------------------------------------------------------------------

def _lmg_key(driver, pwm, coarse, fine, temp):
    return f'{driver}/{pwm}/{coarse}-{fine}/{temp}C'


def _kapu_key(driver, pwm, mode, temp):
    return f'{driver}/{pwm}/{mode}/{temp}C'


# ---------------------------------------------------------------------------
# SingleTDCData
# ---------------------------------------------------------------------------

class SingleTDCData:
    """
    Load and process TDC time-profile data for one hemisphere/emitter/temperature.

    Parameters
    ----------
    hemisphere  : str    e.g. '56'
    pwm         : int    power setting
    temp        : int    temperature [°C]
    diode       : str    e.g. 'LMG405'
    coarse      : int    LMG coarse setting
    fine        : int    LMG fine setting
    mode        : str    KAPU mode
    double_peak : bool   use double-peak fit model
    uncertainty : bool   use Poissonian bin uncertainties in fit
    base_path   : str    path template with {hem} placeholder

    Attributes (selected)
    ---------------------
    fwhm, fwhm_std          : FWHM and MC uncertainty [ns]
    pocam_fwhm, pocam_fwhm_err : corrected FWHM and total error [ns]
    fit, cov                : fit parameters and covariance
    hist_counts_check       : histogram counts (check binning)
    histogram_check_edges   : bin edges for check histogram
    hist_timestamps_check   : bin midpoints for check histogram
    meas_time               : UTC Unix timestamp
    date                    : compact date string 'YYYYMMDD'
    target                  : '1' or '2'
    diode, pwm, coarse, fine, mode, temp : as passed
    second_peak_contr       : fractional contribution of 2nd peak (double peak only)
    second_peak_fwhm        : FWHM of 2nd peak [ns]         (double peak only)
    second_peak_mean        : mean of 2nd peak [ns]          (double peak only)
    second_peak_delay       : delay of 2nd peak vs main [ns] (double peak only)
    """

    def __init__(self, hemisphere, pwm, temp, diode,batch,
                 coarse=1, fine=20, mode='default',
                 double_peak=False, uncertainty=True,
                 base_path=None):

        self.diode       = diode
        self.hemisphere  = hemisphere
        self.pwm         = pwm
        self.coarse      = coarse
        self.fine        = fine
        self.mode        = mode
        self.temp        = temp
        self.double_peak = double_peak
        self.uncertainty = uncertainty
        self.batch       = batch

        h5_path = base_path.format(batch = batch, hem=hemisphere)
        h = h5.File(h5_path + diode, 'r')

        # Try target '1', fall back to '2'
        for t in ('1', '2'):
            prefix = 'lmg' if 'LMG' in diode else 'kapu'
            driver = prefix + t
            try:
                if 'LMG' in diode:
                    key = _lmg_key(driver, pwm, coarse, fine, temp)
                else:
                    key = _kapu_key(driver, pwm, mode, temp)
                self.times_ps = np.array(h[key].get('time_profile'))
                self.mean_occ = float(h[key + '/metadata'].attrs.get('mean_occ'))
                self.target   = t
                self.driver   = driver
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

        # ---- Histogramming ----
        times_ns = self.times_ps / 1000.0

        prelim = np.histogram(times_ns, 100)
        approx_peak_time = prelim[1][np.argmax(prelim[0])]

        # Working histogram (for fitting)
        self.histogram = self._silent_hist(
            times_ns,
            np.linspace(approx_peak_time - 60, approx_peak_time + 90, 150))

        self.hist_counts     = self.histogram[0]
        self.hist_timestamps = self._get_mids(self.histogram)

        self.corrected_hist_counts, self.correction_factors = \
            poissonian_correction(self.mean_occ, self.hist_counts)

        self.peak       = np.max(self.hist_counts)
        peak_idx        = np.argmax(self.hist_counts)
        self.peak_time  = self.hist_timestamps[peak_idx]

        # Interpolation for approximate FWHM seed
        interp = scipy.interpolate.InterpolatedUnivariateSpline(
            self.hist_timestamps, self.hist_counts)
        t_fine  = np.linspace(self.peak_time - 30, self.peak_time + 30, 100)
        y_fine  = interp(t_fine)

        left_idx  = np.argmin(np.abs(y_fine[:len(t_fine[t_fine < self.peak_time])]
                                     - 0.5 * self.peak))
        right_idx = np.argmin(np.abs(
            y_fine[-len(t_fine[t_fine > self.peak_time]):]
            - 0.5 * self.peak))

        approx_width = (t_fine[t_fine > self.peak_time][right_idx]
                        - t_fine[t_fine < self.peak_time][left_idx])

        if self.uncertainty:
            self.hist_counts_err = np.sqrt(np.clip(self.hist_counts, 10, None))

        # ---- Fitting ----
        if not double_peak:
            fit_func    = skewed_gaussian
            param_label = '[mean, sigma, scaling, alpha]'
            p0_base     = [approx_peak_time, approx_width / 2,
                           np.max(self.hist_counts), 0]
        else:
            fit_func    = skewed_double_gaussian
            param_label = '[mean, sigma, scaling, alpha, mean_2, sigma_2, scaling_2]'
            p0_base     = [approx_peak_time, approx_width,
                           np.max(self.hist_counts), 0,
                           approx_peak_time + 10, approx_width * 2 / 3,
                           np.max(self.hist_counts) / 7]

        # Select best initial alpha
        best_alpha, best_trace = 0, np.inf
        for alpha in [0, 1, 5, 10]:
            p0 = list(p0_base)
            p0[3] = alpha
            try:
                cov = curve_fit(fit_func, self.hist_timestamps,
                                self.hist_counts, p0=p0,
                                maxfev=10000)[1]
                trace = sum(cov[i][i] for i in range(3))
                if trace < best_trace:
                    best_trace, best_alpha = trace, alpha
            except RuntimeError:
                continue

        p0_best    = list(p0_base)
        p0_best[3] = best_alpha

        fit_kwargs = dict(p0=p0_best, maxfev=10000)
        if self.uncertainty:
            fit_kwargs.update(sigma=self.hist_counts_err, absolute_sigma=True)

        self.fit, self.cov = curve_fit(
            fit_func, self.hist_timestamps, self.hist_counts, **fit_kwargs)

        self.fit_function = lambda x: fit_func(x, *self.fit)
        self.norm = 1.0 / np.sum(self.hist_counts)

        # ---- Broadening correction ----
        sigma_eff        = self.fit[1]
        sigma_eff_std    = np.sqrt(abs(self.cov[1][1]))
        sigma_pocam      = np.sqrt(sigma_eff ** 2 - DELTA_BROADENING ** 2)
        self.fit[1]      = sigma_pocam
        self.cov[1][1]   = sigma_eff / sigma_pocam * sigma_eff_std ** 2

        # ---- FWHM via MC ----
        x_range = np.linspace(90, 140, 15000)
        self.fwhm, self.fwhm_std = compute_fwhm_mc(
            fit_func, self.fit, self.cov, x_range)

        self.pocam_fwhm     = round(self.fwhm, 2)
        self.pocam_fwhm_err = round(np.sqrt(self.fwhm_std ** 2
                                            + 2 * DELTA_BROADENING ** 2), 2)

        # ---- Check histogram (for output) ----
        check_bins = np.linspace(self.peak_time - 60,
                                 self.peak_time + 90, 150, endpoint=False)
        check_hist = np.histogram(times_ns, bins=check_bins)
        self.hist_counts_check      = check_hist[0]
        self.histogram_check_edges  = check_hist[1]
        self.hist_timestamps_check  = self._get_mids(check_hist)

        # ---- Double-peak extras ----
        if double_peak:
            mean_2  = self.fit[4]
            sigma_2 = self.fit[5]
            mask    = np.abs(self.hist_timestamps - mean_2) < 2.3548 * abs(sigma_2)
            self.second_peak_contr = round(
                np.sum(self.hist_counts[mask]) / np.sum(self.hist_counts), 4)
            self.second_peak_fwhm  = round(2.3548 * abs(sigma_2), 2)
            self.second_peak_mean  = round(mean_2, 2)
            self.first_peak_mean   = round(self.fit[0], 2)
            self.second_peak_delay = round(
                self.second_peak_mean - self.first_peak_mean, 2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mids(histogram):
        edges = histogram[1]
        return 0.5 * (edges[:-1] + edges[1:])

    @staticmethod
    def _silent_hist(data, bins):
        """Create a histogram without rendering a matplotlib figure."""
        return np.histogram(data, bins=bins)


# ---------------------------------------------------------------------------
# JSON assembly and file writing
# ---------------------------------------------------------------------------

def _driver_char(diode):
    return 'l' if 'LMG' in diode else 'k'


def compute_and_save(hemisphere, device_id, emitter,
                     pwm, temp, coarse, fine, mode,
                     double_peak, uncertainty,
                     paths, batch):
    """
    Compute TDC time profile and write JSON output file.

    Parameters
    ----------
    hemisphere  : str   HDF5 hemisphere id
    device_id   : str   POCAM device number e.g. '016'
    emitter     : str   e.g. 'LMG405'
    double_peak : bool  use double-peak fit model
    uncertainty : bool  use Poissonian bin weights in fit
    paths       : dict  with keys 'data', 'output'

    Returns
    -------
    dict — assembled time-profile dictionary
    """
    base_path = paths['data']
    driver    = _driver_char(emitter)
    batch = batch

    tdc = SingleTDCData(
        hemisphere=hemisphere, pwm=pwm, temp=temp, diode=emitter,
        coarse=coarse, fine=fine, mode=mode, batch = batch,
        double_peak=double_peak, uncertainty=uncertainty,
        base_path=base_path)

    target_label = 'master' if tdc.target == '1' else 'slave'

    # Fit scan for output
    x_fit_scan = np.linspace(tdc.peak_time - 60, tdc.peak_time + 90,
                              1500, endpoint=False)
    # Adjust for offset so pulse starts near 100 ns
    offset = tdc.hist_timestamps_check[tdc.hist_counts_check > 20][0]
    adjust = offset - 100
    x_fit_scan_adj = x_fit_scan - adjust
    fit_values     = tdc.norm * tdc.fit_function(x_fit_scan + adjust)

    # ---- hist-with-fit entry ----
    hist_entry = {
        'data_format': 'hist-with-fit',
        'power':        pwm,
        'temperature':  temp,
        'x_label':      'Time [ns]',
        'y_label':      'Relative Counts',
        'y_values':     (tdc.hist_counts_check / 9000).tolist(),
        'x_min':        float(np.min(tdc.histogram_check_edges)),
        'x_max':        float(np.max(tdc.histogram_check_edges)),
        'n_bins':       len(tdc.histogram_check_edges) - 1,
        'fit_x_min':    float(np.min(x_fit_scan_adj)),
        'fit_x_max':    float(np.max(x_fit_scan_adj)),
        'fit_n_bins':   len(x_fit_scan_adj),
        'fit_y_values': fit_values.tolist(),
        'title':        'Time Profile of Single Emitted Pulses',
    }
    if 'LMG' in emitter:
        hist_entry['coarse'] = coarse
        hist_entry['fine']   = fine
    else:
        hist_entry['mode'] = mode

    # ---- FWHM value entry ----
    fwhm_entry = {
        'data_format': 'value',
        'value':        tdc.pocam_fwhm,
        'error':        tdc.pocam_fwhm_err,
        'power':        pwm,
        'temperature':  temp,
        'label':        'FWHM',
        'title':        'Full-Width-Half-Maximum [ns]',
    }
    if 'LMG' in emitter:
        fwhm_entry['coarse'] = coarse
        fwhm_entry['fine']   = fine
    else:
        fwhm_entry['mode'] = mode

    # ---- Comments ----
    comments = [
        "The data shows the reconstructed time profile of a single light pulse "
        "for the chosen L(E)D at the specified conditions.",
        "Experimental data: normalized histogram of photon arrival timestamps.",
        "coarse/fine (LMG) or mode (KAPU): pulse-shape settings.",
        "power: PWM value (0–54000 ≈ 0–32 V).",
        "Fit function: skewed Gaussian × optional second plain Gaussian.",
        "FWHM stored as value including MC uncertainty; refers to the main peak.",
        "Time axis shifted so pulses start near 100 ns (internal POCAM delay).",
        "Structure of device_uid: 'pocam-{date}_{device_number}'",
        "Structure of subdevice_uid: 'pocam-led-{target}_{driver}-{wl}_{device}'",
    ]
    if double_peak:
        comments.append(
            f"Second peak is ~{tdc.second_peak_delay} ns delayed from main peak, "
            f"FWHM ~{tdc.second_peak_fwhm} ns, "
            f"contribution ~{round(100 * tdc.second_peak_contr, 2)} %.")

    result = {
        'device_uid':    f'pocam-{tdc.date}_{device_id}',
        'subdevice_uid': f'pocam-led-{target_label}_{driver}-{emitter[-3:]}_{device_id}',
        'meas_name':     'pulse-time-profile',
        'meas_class':    'display',
        'meas_stage':    'calibration',
        'meas_group':    'timing',
        'meas_site':     'tum',
        'meas_time':     tdc.meas_time,
        'meas_data':     [hist_entry, fwhm_entry],
        'comments':      comments,
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
                'pathname': '/data/exp/IceCubeUpgrade/commissioning/pocam/POCAM_documentation.pdf',
                'comment':  'POCAM documentation guide.',
            },
        ],
    }

    if "LMG" in emitter:
        fname = f'tdc_{pwm}_{temp}C_{coarse}-{fine}.json'
    if "KAPU" in emitter:
        fname = f'tdc_{pwm}_{temp}C_{mode}.json'
        
    out_dir = os.path.join(paths['output'], f'{batch}',
                                f'pocam_{device_id}',
                                f'hem_{hemisphere}',
                                f'{emitter}')
    
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)  
    out_path = os.path.join(out_dir, fname)
    
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f'  Saved → {out_path}')

    return result