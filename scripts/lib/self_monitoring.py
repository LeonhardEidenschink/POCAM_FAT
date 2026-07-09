"""
self_monitoring.py
==================
Library for POCAM self-monitoring, intensity (PD / SiPM), and fit-parameter
calculations.

Contains:
  - read_self_monitoring_file        : parse one self-monitoring HDF5 file
  - cross_check_absolute_calibration : pull (pwm, value) pairs from abs-cal JSONs
  - self_monitoring_analysis         : PD linear fit + SiPM spline/isotonic fit
  - compute_and_save                 : top-level routine — runs the analysis for
                                        one hemisphere / emitter / temperature
                                        combination and writes a JSON output file

Import this from your run script — do not run directly.
"""

import glob
import json
import os

import h5py as h5
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression

# FIXED: this import was dropped in the previous edit, but extract_per_pwm is
# actually used below in read_self_monitoring_file. Re-added.
from pcm_monitoring_func import extract_per_pwm


def linear(x, a, b):
    return a * x + b


def _to_jsonable(obj):
    """Recursively convert numpy types/arrays to plain Python so json.dump works."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def read_self_monitoring_file(filename):
    """
    Read a POCAM self-monitoring HDF5 file and extract PD/SiPM data for every
    flasher into a nested dictionary.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        data_dict[flasher]['pd_data']['adcA' | 'adcB'][...]
        data_dict[flasher]['sipm_data']['tdc2'][...]
    """
    f_ = h5.File(filename, 'r')

    all_data = {}
    for sipm_pwm_ in f_['metadata/sipm_pwm']:
        all_data[sipm_pwm_] = {}
        for flm_ in f_['metadata']['flasher_modes']:
            all_data[sipm_pwm_][flm_.decode()] = extract_per_pwm(
                f_['flashes/{:d}/{:s}'.format(sipm_pwm_, flm_.decode())]
            )

    data_dict = {}
    flasher_pwm = f_['metadata']['flasher_pwm'][()]
    lmg1_app = '_A'

    for flasher in ['lmg0', 'lmg1', 'lmg2', 'lmg3', 'kapu0', 'kapu3']:
        data_dict[flasher] = {}
        data_dict[flasher]["sipm_data"] = {}
        data_dict[flasher]["pd_data"] = {}

        flasher_pd = flasher + lmg1_app if 'lmg1' in flasher else flasher

        for sipm_pwm_ in f_['metadata']['sipm_pwm'][()]:
            plotval_a = np.array([
                all_data[sipm_pwm_][flasher_pd][fpwm_]['adcA']['sum'].mean()
                - all_data[sipm_pwm_][flasher_pd][0]['adcA']['sum'].mean()
                for fpwm_ in flasher_pwm
            ])
            data_dict[flasher]["pd_data"]['adcA'] = {
                'flasher_pwm':      flasher_pwm,
                'adcA_peak_mean':   plotval_a,
                'adcA_peak_std':    np.array([
                    all_data[sipm_pwm_][flasher_pd][fpwm_]['adcA']['peak'].std()
                    for fpwm_ in flasher_pwm
                ]),
            }

            plotval_b = np.array([
                all_data[sipm_pwm_][flasher_pd][fpwm_]['adcB']['sum'].mean()
                - all_data[sipm_pwm_][flasher_pd][0]['adcB']['sum'].mean()
                for fpwm_ in flasher_pwm
            ])
            data_dict[flasher]["pd_data"]['adcB'] = {
                'flasher_pwm':      flasher_pwm,
                'adcB_peak_mean':   plotval_b,
                'adcB_peak_std':    np.array([
                    all_data[sipm_pwm_][flasher_pd][fpwm_]['adcB']['peak'].std()
                    for fpwm_ in flasher_pwm
                ]),
            }

            plotval_tdc2 = np.array([
                all_data[sipm_pwm_][flasher_pd][fpwm_]['tdc2_dt'].mean()
                for fpwm_ in flasher_pwm
            ])
            data_dict[flasher]["sipm_data"]['tdc2'] = {
                'flasher_pwm':      flasher_pwm,
                'tdc2_peak_mean':   plotval_tdc2,
                'tdc2_peak_std':    np.array([
                    all_data[sipm_pwm_][flasher_pd][fpwm_]['tdc2_dt'].std()
                    for fpwm_ in flasher_pwm
                ]),
            }

    return data_dict


def cross_check_absolute_calibration(abs_cal_glob):
    """Read (pwm, value) pairs from all abs-cal JSON files matching abs_cal_glob."""
    pwm = []
    val_pd = []
    filelist = glob.glob(abs_cal_glob)
    if not filelist:
        raise FileNotFoundError(
            f'No absolute-calibration files matched pattern: {abs_cal_glob}'
        )
    for file in filelist:
        with open(file, 'r') as f:
            data = json.load(f)
            pwm.append(data['meas_data'][0]['power'])
            val_pd.append(data['meas_data'][0]['value'])

    sorted_idx = np.argsort(pwm)
    pwm = np.array(pwm)[sorted_idx]
    val_pd = np.array(val_pd)[sorted_idx]

    return pwm, val_pd


def self_monitoring_analysis(emitter, flasher_key, sm_filepath, abs_cal_glob,
                              smoothing_factor=1.0):
    """
    Load one self-monitoring HDF5 file, cross-check against absolute
    calibration, and return PD linear-fit and SiPM spline/isotonic fit
    parameters.

    Parameters
    ----------
    emitter          : str   device model name, e.g. 'LMG405' — used only to
                             look up ref_photons and for labeling. NOT a key
                             into the HDF5 data.
    flasher_key      : str   hardware flasher-slot key as stored in the
                             self-monitoring HDF5 file, e.g. 'lmg1', 'kapu0'.
                             This is DIFFERENT from `emitter`: which physical
                             LED model sits in which slot varies per
                             hemisphere/device, so this must be resolved by
                             the caller (your original pocam_utils helpers
                             _lmg_key / _kapu_key / _resolve_target look like
                             they did exactly this — I don't have their
                             implementation, so I can't reproduce it here).
    sm_filepath      : str   path to the self-monitoring HDF5 file
    abs_cal_glob     : str   glob pattern matching absolute-calibration JSON files
    smoothing_factor : float scales the UnivariateSpline smoothing term `s`
    """
    ref_photons = {
        'LMG365':  2.0e7,
        'LMG405':  1.0e9,
        'LMG450':  1.0e9,
        'LMG520':  1.0e9,
        'KAPU405': 1.0e8,
        'KAPU465': 1.0e8,
    }

    data_dict = read_self_monitoring_file(sm_filepath)
    pwm, val_pd = cross_check_absolute_calibration(abs_cal_glob)

    mask_pd = np.isin(data_dict[flasher_key]["pd_data"]['adcA']['flasher_pwm'], pwm)
    n_pd = data_dict[flasher_key]["pd_data"]['adcA']['adcA_peak_mean'][mask_pd]

    mask_sipm = np.isin(data_dict[flasher_key]["sipm_data"]['tdc2']['flasher_pwm'], pwm)
    n_sipm = data_dict[flasher_key]["sipm_data"]['tdc2']['tdc2_peak_mean'][mask_sipm]

    fit_results = {}

    # --- PD: linear fit ---
    popt_pd, pcov_pd = curve_fit(linear, val_pd, n_pd)
    fit_results['pd'] = {
        'pwm':              pwm,
        'val_pd':           val_pd,
        'n_pd':             n_pd,
        'slope':            popt_pd[0],
        'intercept':        popt_pd[1],
        'slope_error':      np.sqrt(pcov_pd[0][0]),
        'intercept_error':  np.sqrt(pcov_pd[1][1]),
        'covariance':       pcov_pd,
    }

    # --- SiPM: log-x spline + isotonic regression ---
    eps = 1e-9
    logx = np.log(val_pd + eps)
    #sort x and y 
    logx, n_sipm = zip(*sorted(zip(logx, n_sipm)))
    spline = UnivariateSpline(
        logx, n_sipm, s=len(logx) * np.var(n_sipm) * smoothing_factor
    )

    x_fine = np.linspace(val_pd.min(), val_pd.max(), 300)
    logx_fine = np.log(x_fine + eps)
    y_spline = spline(logx_fine)

    iso = IsotonicRegression(increasing=True)
    y_fine = iso.fit_transform(x_fine, y_spline)

    reference_value = np.interp(ref_photons[emitter], x_fine, y_fine)

    fit_results['sipm'] = {
        'pwm':              pwm,
        'val_pd':           val_pd,
        'n_sipm':           n_sipm,
        'x_fine':           x_fine,
        'y_fine':           y_fine,
        'ref_photons':      ref_photons[emitter],
        'reference_value':  reference_value,
    }

    return fit_results


# ---------------------------------------------------------------------------
# compute_and_save
# ---------------------------------------------------------------------------

def compute_and_save(hemisphere, device_id, emitter, flasher_key, temp,
                      sm_filepath, abs_cal_glob, output_dir, batch,
                      settings=None, smoothing_factor=1.0):
    """
    Run the self-monitoring fit analysis for one hemisphere / emitter /
    temperature combination and write the fit results to a JSON file.

    Parameters
    ----------
    hemisphere    : str   HDF5 hemisphere id, e.g. '03'
    device_id     : str   POCAM device number, e.g. '002'
    emitter       : str   device model name, e.g. 'LMG405' (for labeling and
                          ref_photons lookup only)
    flasher_key   : str   hardware flasher-slot key in the HDF5 file, e.g.
                          'lmg1' — see self_monitoring_analysis() docstring;
                          this must be resolved by the caller, it is NOT
                          derived from `emitter` automatically.
    temp          : int   temperature [°C]
    sm_filepath   : str   path to the self-monitoring HDF5 file for this run
    abs_cal_glob  : str   glob pattern matching absolute-calibration JSON files
    output_dir    : str   base output directory (JSON is written under
                          {output_dir}/{batch}/pocam_{device_id}/hem_{hemisphere}/{emitter}/)
    batch         : str   e.g. 'batch2'
    settings      : dict or None
                    fixed measurement settings from config (pwm, coarse, fine,
                    mode, target) — recorded as metadata alongside the fit
                    results for traceability. NOTE: these values are NOT used
                    in the fit computation itself (pwm/power values are read
                    from the abs-cal files, not from this dict) — verify that
                    this is the behavior you want; if 'target' should instead
                    select between two hemispheres in the HDF5 file, that
                    logic isn't implemented here and needs to be added.
    smoothing_factor : float   passed through to the SiPM spline fit

    Returns
    -------
    dict — the same dict that gets written to disk
    """
    settings = settings or {}

    fit_results = self_monitoring_analysis(
        emitter, flasher_key, sm_filepath, abs_cal_glob,
        smoothing_factor=smoothing_factor,
    )

    pd_entry = {
        'data_format':      'fit_linear',
        'label':            'PD_vs_power',
        'title':            'PD signal vs. flasher power — linear fit',
        'power':            fit_results['pd']['pwm'],
        'value':            fit_results['pd']['n_pd'],
        'slope':            fit_results['pd']['slope'],
        'intercept':        fit_results['pd']['intercept'],
        'slope_error':      fit_results['pd']['slope_error'],
        'intercept_error':  fit_results['pd']['intercept_error'],
        'covariance':       fit_results['pd']['covariance'],
        'temperature':      temp,
    }

    sipm_entry = {
        'data_format':      'fit_spline_isotonic',
        'label':            'SiPM_vs_power',
        'title':            'SiPM signal vs. abs-cal PD value — spline + isotonic fit',
        'power':            fit_results['sipm']['pwm'],
        'val_pd':           fit_results['sipm']['val_pd'],
        'value':            fit_results['sipm']['n_sipm'],
        'x_fine':           fit_results['sipm']['x_fine'],
        'y_fine':           fit_results['sipm']['y_fine'],
        'ref_photons':      fit_results['sipm']['ref_photons'],
        'reference_value':  fit_results['sipm']['reference_value'],
        'temperature':      temp,
    }

    meas_data = [pd_entry, sipm_entry]

    comments = [
        'PD entry: linear fit of PD picoamp signal vs. flasher power setting.',
        'SiPM entry: log-x smoothing spline + isotonic regression of SiPM '
        'signal vs. absolute-calibration PD value; reference_value is the '
        'isotonic curve evaluated at ref_photons for this emitter.',
        f'Configured settings (metadata only, not used in fit): {settings}',
    ]

    result = {
        'device_uid':    f'pocam-{device_id}',
        'emitter':       emitter,
        'hemisphere':    hemisphere,
        'batch':         batch,
        'settings':      settings,
        'meas_name':     'self-monitoring-fits',
        'meas_class':    'display',
        'meas_stage':    'calibration',
        'meas_group':    'intensity',
        'meas_site':     'tum',
        'meas_data':     meas_data,
        'comments':      comments,
    }

    out_dir = os.path.join(
        output_dir, batch, f'pocam_{device_id}', f'hem_{hemisphere}', emitter
    )
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    filename = f'self_monitoring_{temp}C.json'
    out_path = os.path.join(out_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(_to_jsonable(result), f, indent=4)
    print(f'  Saved -> {out_path}')

    return result