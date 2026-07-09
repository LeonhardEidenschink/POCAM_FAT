"""
self_monitoring.py
==================
Library for POCAM self-monitoring, intensity (PD / PMT), and photon-number
calculations.

Contains:
  - SinglePDData          : load and pre-process one PD picoamp dataset
  - SinglePMTData         : load and pre-process one PMT waveform dataset
  - SingleAngularCalData  : load flange-calibration data and integrate the
                            angular emission pattern
  - compute_and_save      : top-level routine — runs everything for one
                            hemisphere / emitter / temperature combination
                            and writes a JSON output file

Import this from your run script — do not run directly.
"""

import json
import os

import h5py as h5
import numpy as np
import scipy.integrate
import scipy.interpolate
from numpy.random import multivariate_normal
from scipy.optimize import curve_fit
import healpy as hp
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression


from pocam_utils import (
    X_PRE, 
    Y_PRE,
    X_VALUES, 
    Y_VALUES,
    NIST,
    sigmoid,
    int_func,
    integrate_solid_angle,
    calc_photons,
    fit_correction_curves,
    _lmg_key,
    _kapu_key,
    _build_key,
    _resolve_target,
)

from POCAMSiPMHandler import breakdown_func, baseline, ampl

from pcm_monitoring_func import decode_value, get_timestamps, getADCreadings, extract_per_pwm, convert_to_perTrig, get_SiPM_dt

def linear(x, a, b):
    return a*x + b

def read_self_monitoring_file(filename):
    """
    Read a POCAM self-monitoring HDF5 file and extract all data into a dictionary.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        Nested dictionary containing all extracted data.
    """
    f_ = h5.File(filename, 'r')


    all_data = {}
    for sipm_pwm_ in f_['metadata/sipm_pwm']:
        all_data[sipm_pwm_]= {}
        for flm_ in f_['metadata']['flasher_modes']:
            all_data[sipm_pwm_][flm_.decode()] = extract_per_pwm(f_['flashes/{:d}/{:s}'.format(sipm_pwm_, flm_.decode())])

    data_dict = {}


    for flasher in ['lmg0', 'lmg1', 'lmg2', 'lmg3', 'kapu0', 'kapu3']:
    data_dict[flasher] = {}
    data_dict[flasher]["sipm_data"] = {}
    data_dict[flasher]["pd_data"] = {}
    
    lmg1_app = '_A'
    
    for sipm_pwm_ in f_['metadata']['sipm_pwm'][()]:#[:2]: 
        if not 'lmg1' in flasher:
            flasher_pd = flasher
        elif 'lmg1' in flasher:
            flasher_pd = flasher + lmg1_app#flasher_pd
        
        
        plotval = np.array([(all_data[sipm_pwm_][flasher_pd][fpwm_]['adcA']['sum'].mean() - all_data[sipm_pwm_][flasher_pd][0]['adcA']['sum'].mean()) for fpwm_ in f_['metadata']['flasher_pwm'][()] ] )

        data_dict[flasher]["pd_data"]['adcA'] = {}
        data_dict[flasher]["pd_data"]['adcA']['flasher_pwm'] = f_['metadata']['flasher_pwm'][()]
        data_dict[flasher]["pd_data"]['adcA']['adcA_peak_mean'] = plotval
        data_dict[flasher]["pd_data"]['adcA']['adcA_peak_std'] = np.array([all_data[sipm_pwm_][flasher_pd][fpwm_]['adcA']['peak'].std() for fpwm_ in f_['metadata']['flasher_pwm'][()] ] )

        plotval = np.array([(all_data[sipm_pwm_][flasher_pd][fpwm_]['adcB']['sum'].mean() - all_data[sipm_pwm_][flasher_pd][0]['adcB']['sum'].mean()) for fpwm_ in f_['metadata']['flasher_pwm'][()] ] )
        
        data_dict[flasher]["pd_data"]['adcB'] = {}
        data_dict[flasher]["pd_data"]['adcB']['flasher_pwm'] = f_['metadata']['flasher_pwm'][()]
        data_dict[flasher]["pd_data"]['adcB']['adcB_peak_mean'] = plotval
        data_dict[flasher]["pd_data"]['adcB']['adcB_peak_std'] = np.array([all_data[sipm_pwm_][flasher_pd][fpwm_]['adcB']['peak'].std() for fpwm_ in f_['metadata']['flasher_pwm'][()] ] )

        plotval = np.array([all_data[sipm_pwm_][flasher_pd][fpwm_]['tdc2_dt'].mean() for fpwm_ in f_['metadata']['flasher_pwm'][()] ] )
        data_dict[flasher]["sipm_data"]['tdc2'] = {}
        data_dict[flasher]["sipm_data"]['tdc2']['flasher_pwm'] = f_['metadata']['flasher_pwm'][()]
        data_dict[flasher]["sipm_data"]['tdc2']['tdc2_peak_mean'] = plotval
        data_dict[flasher]["sipm_data"]['tdc2']['tdc2_peak_std'] = np.array([all_data[sipm_pwm_][flasher_pd][fpwm_]['tdc2_dt'].std() for fpwm_ in f_['metadata']['flasher_pwm'][()] ] )

        return data_dict

def cross_check_absolute_calibration(hemisphere, device_id, emitter, path_abs_cal):
    pwm = []
    val_pd = []
    filelist = glob.glob(path_abs_cal)
    for file in filelist:
        with open(file, 'r') as f:
            data = json.load(f)
            pwm.append(data['meas_data'][0]['power'])
            val_pd.append(data['meas_data'][0]['value'])

    sorted_idx = np.argsort(pwm)
    pwm  = np.array(pwm)[sorted_idx]
    val_pd = np.array(val_pd)[sorted_idx]

    return pwm, val_pd

def self_monitoring_analysis(hemisphere, device_id, emitter, temp, paths, batch):

    filepath = 
    data_dict = read_self_monitoring_file(filepath)

    pwm, val_pd = cross_check_absolute_calibration(hemisphere, device_id, emitter, paths['abs_cal_path'])

    mask_pd = np.isin(data_dict[emitter]["pd_data"]['adcA']['flasher_pwm'], pwm)
    n_pd = data_dict[emitter]["pd_data"]['adcA']['adcA_peak_mean'][mask_pd]

    mask_sipm = np.isin(data_dict[emitter]["sipm_data"]['tdc2']['flasher_pwm'], pwm)
    n_sipm = data_dict[emitter]["sipm_data"]['tdc2']['tdc2_peak_mean'][mask_sipm]

    fit_results = {}
    #make a linear fit of the pd data and save the fit parameters and the errors in a dictionary
    popt_pd, pcov_pd = curve_fit(linear, pwm, n_pd)
    fit_results['pd'] = {'pwm': pwm,
                        'n_pd': n_pd, 
                        'slope': popt_pd[0],
                        'intercept': popt_pd[1],
                        'slope_error': np.sqrt(pcov_pd[0][0]),
                        'intercept_error': np.sqrt(pcov_pd[1][1]),
                        'covariance': pcov_pd,
                        }
    
    #for the sipm data, just spline the data with a logaritmic spline, the data looks like a logarithm so it makes sense 
    # then, save the value of the spline ar a reference value that I will define somewhere else
    eps = 1e-9
    logx = np.log(val_pd + eps)

    spline = UnivariateSpline(logx, n_sipm,
                                   s=len(logx) * np.var(n_sipm) * SMOOTHING_FACTOR)

    # evaluate on a fine linear-x grid, transformed into log-space
    x_fine = np.linspace(val_pd.min(), val_pd.max(), 300)
    logx_fine = np.log(val_pd + eps)
    y_spline = spline(logx_fine)

    iso = IsotonicRegression(increasing=True)
    y_fine = iso.fit_transform(x_fine, y_spline)

    





# ---------------------------------------------------------------------------
# compute_and_save
# ---------------------------------------------------------------------------

def compute_and_save(hemisphere, device_id, emitter,
                     temp,
                     paths, batch,
                     target=None):
    """
    Compute self-monitoring intensity and photon number for one combination,
    then write a JSON output file.

    Parameters
    ----------
    hemisphere : str   HDF5 hemisphere id  e.g. '03'
    device_id  : str   POCAM device number e.g. '02'
    emitter    : str   e.g. 'LMG405'
    pwm        : int   power setting
    temp       : int   temperature [°C]
    coarse     : int   LMG coarse setting
    fine       : int   LMG fine setting
    mode       : str   KAPU mode
    paths      : dict  keys: 'data', 'output', 'cal_prefix'
    batch      : str   e.g. 'batch2'
    target     : str or None   '1' or '2'; None = auto-detect from HDF5

    Returns
    -------
    dict — assembled result dictionary
    """
    base_path  = paths['data']
    cal_prefix = paths.get('cal_prefix', 'cali_flange_')

    # ---- Auto-detect target if not specified ----
    if target is None:
        h5_path = base_path.format(batch =batch, hem=hemisphere)
        print(h5_path)
        h       = h5.File(h5_path + emitter, 'r')
        target, _ = _resolve_target(h, emitter, pwm, coarse, fine, mode, temp)
        print(target)
        h.close()

    target_label = 'master' if target == '1' else 'slave'

    # ---- PD signal at requested conditions ----
    pd_data = SinglePDData(
        hemisphere=hemisphere, pwm=pwm, temp=temp, diode=emitter,
        target=target, coarse=coarse, fine=fine, mode=mode, batch=batch,
        base_path=base_path)
    print(pd_data)

    # ---- PD signal at baseline conditions (25 °C, max power, default shape) ----
    pd_norm = SinglePDData(
        hemisphere=hemisphere, pwm=54000, temp=25, diode=emitter,
        target=target, coarse=1, fine=20, mode='default', batch=batch,
        base_path=base_path)

    # ---- Angular calibration + baseline photon number ----
    cal = SingleAngularCalData(
        hemisphere=hemisphere, diode=emitter, batch=batch,
        base_path=base_path, cal_prefix=cal_prefix)

    # ---- Scale photons from PD ratio ----
    pd_ratio           = pd_data.mean_signal_vals / pd_norm.mean_signal_vals
    emitted_photons_pd = max(0.0, pd_ratio * cal.photons)

    pd_rel_err   = pd_data.mean_signal_err  / pd_data.mean_signal_vals
    norm_rel_err = pd_norm.mean_signal_err  / pd_norm.mean_signal_vals

    pd_rel_total = np.sqrt(pd_rel_err**2
                           + norm_rel_err**2
                           + cal.rel_stat_error**2
                           + cal.rel_sys_error**2)
    pd_mean_rel_total = np.sqrt((pd_rel_err / np.sqrt(100))**2
                                + (norm_rel_err / np.sqrt(100))**2
                                + cal.mean_rel_stat_error**2
                                + cal.mean_rel_sys_error**2)

    # ---- PMT cross-calibration (best-effort) ----
    emitted_photons_pmt = None
    try:
        pmt_data = SinglePMTData(
            hemisphere=hemisphere, pwm=pwm, temp=temp, diode=emitter,
            target=target, coarse=coarse, fine=fine, mode=mode, batch=batch,
            base_path=base_path)
        data_pmt_mean = float(np.mean(pmt_data.processed_data['integrated_signal']))

        # Find highest PMT power that stays below saturation (<4500 mV peak)
        norm_pwm = 54000
        for try_pwm in [54000, 45000, 35000, 30000, 25000, 20000, 15000, 10000, 7500]:
            norm_pmt_obj = SinglePMTData(
                hemisphere=hemisphere, pwm=try_pwm, temp=temp, diode=emitter, batch=batch,
                target=target, coarse=1, fine=20, mode='default',
                base_path=base_path)
            if np.max(norm_pmt_obj.processed_data['peak']) < 4500:
                norm_pwm = try_pwm
                break
        norm_pmt_mean = float(np.mean(norm_pmt_obj.processed_data['integrated_signal']))

        norm_pd_cross = SinglePDData(
            hemisphere=hemisphere, pwm=norm_pwm, temp=temp, diode=emitter,
            target=target, coarse=coarse, fine=fine, mode=mode, batch=batch,
            base_path=base_path).mean_signal_vals

        emitted_photons_pmt = ((data_pmt_mean / norm_pmt_mean)
                               * (norm_pd_cross / pd_norm.mean_signal_vals)
                               * cal.photons)

    except Exception as e:
        print(f'  PMT cross-calibration skipped — {e}')

    # ---- Assemble result entries ----
    pd_entry = {
        'data_format': 'value',
        'value':        round(emitted_photons_pd),
        'error':        round(emitted_photons_pd * pd_rel_total),
        'rel_error':    round(pd_rel_total, 4),
        'power':        pwm,
        'temperature':  temp,
        'label':        'N_photons_PD',
        'title':        'Estimated Number of Emitted Photons per Pulse (PD)',
    }
    if 'LMG' in emitter:
        pd_entry['coarse'] = coarse
        pd_entry['fine']   = fine
    else:
        pd_entry['mode'] = mode

    meas_data = [pd_entry]

    if emitted_photons_pmt is not None:
        pmt_entry = {
            'data_format': 'value',
            'value':        round(emitted_photons_pmt),
            'power':        pwm,
            'temperature':  temp,
            'label':        'N_photons_PMT',
            'title':        'Estimated Number of Emitted Photons per Pulse (PMT cross-cal)',
        }
        if 'LMG' in emitter:
            pmt_entry['coarse'] = coarse
            pmt_entry['fine']   = fine
        else:
            pmt_entry['mode'] = mode
        meas_data.append(pmt_entry)

    comments = [
        'Estimated total number of emitted photons per pulse.',
        'Method: PD picoamp signal ratio × angular-integration baseline.',
        'Baseline derived from flange-calibration HDF5 data.',
        'Air-to-ice correction applied via pre-measured sigmoid model.',
        'Errors: systematic (distance, NIST resp., point-source approx.) '
        'and statistical (MC integration, signal noise, sim. mismatch).',
        f'Hemisphere SN: {cal.hemisphere_sn}',
        f'Pulse duration: {cal.pulse_time_s * 1e6:.3f} µs',
        'Structure of device_uid: pocam-{date}_{device_number}',
    ]

    driver_char = 'l' if 'LMG' in emitter else 'k'
    result = {
        'device_uid':    f'pocam-{device_id}',
        'subdevice_uid': f'pocam-led-{target_label}_{driver_char}-{emitter[-3:]}_{device_id}',
        'meas_name':     'intensity-photon-number',
        'meas_class':    'display',
        'meas_stage':    'calibration',
        'meas_group':    'intensity',
        'meas_site':     'tum',
        'meas_data':     meas_data,
        'comments':      comments,
        'support_files': [
            {
                'filetype': 'hdf5',
                'hostname': 'data.icecube.wisc.edu',
                'pathname': (f'/data/exp/IceCubeUpgrade/commissioning/pocam/'
                             f'pocam_{device_id}/{target_label}_hemisphere/{emitter}'),
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

    out_dir = os.path.join(paths['output'], batch,
                           f'pocam_{device_id}',
                           f'hem_{hemisphere}',
                           emitter)
    os.makedirs(out_dir, exist_ok=True)
    filename = f'self_monitoring_{temp}C.json'
    out_path = os.path.join(out_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f'  Saved → {out_path}')

    return result