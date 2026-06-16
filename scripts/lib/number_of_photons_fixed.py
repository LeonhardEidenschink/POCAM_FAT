"""
photons_lib.py
==============
Library for POCAM number-of-photons calculations.
Contains data loading classes, physics helpers, and the JSON-writing routine.
Import this from your run script — do not run directly.
"""

import json
import os
import numpy as np
import scipy
import scipy.interpolate
import h5py as h5
import healpy as hp
from scipy.optimize import curve_fit

from pocam_utils import (
    sigmoid,
    fit_correction_curves,
    apply_ice_correction,
    datetime_to_unix,
    format_date,
    calc_photons,
    int_func,
    integrate_solid_angle,
    NIST,
    HC,
)

PMT_SATURATION_PEAK = 4500   # [mV]  threshold for PMT saturation check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _driver_key(diode, target):
    """Return the HDF5 driver group key, e.g. 'lmg1' or 'kapu2'."""
    prefix = 'lmg' if 'LMG' in diode else 'kapu'
    return prefix + str(target)


def _lmg_key(driver, pwm, coarse, fine, temp):
    return f'{driver}/{pwm}/{coarse}-{fine}/{temp}C'


def _kapu_key(driver, pwm, mode, temp):
    return f'{driver}/{pwm}/{mode}/{temp}C'


def _lmg_key_precheck(driver, pwm, coarse, fine, temp):
    return f'{driver}/{pwm}/{coarse}-{fine}/{temp}'


def _kapu_key_precheck(driver, pwm, mode, temp):
    return f'{driver}/{pwm}/{mode}/{temp}'


def _data_key(diode, driver, pwm, coarse, fine, mode, temp):
    """Return the correct HDF5 key depending on diode type and temp flag."""
    is_precheck = (temp == '25C_precheck')
    if 'LMG' in diode:
        return (_lmg_key_precheck if is_precheck else _lmg_key)(
            driver, pwm, coarse, fine, temp)
    else:
        return (_kapu_key_precheck if is_precheck else _kapu_key)(
            driver, pwm, mode, temp)


def _driver_char(diode):
    return 'l' if 'LMG' in diode else 'k'


# ---------------------------------------------------------------------------
# PD data loader
# ---------------------------------------------------------------------------

class SinglePDData:
    """
    Load and pre-process one PD picoamp measurement set.

    Tries driver target '1' first, falls back to '2'.

    Attributes
    ----------
    mean_signal_vals : float   background-subtracted mean PD signal [A] (positive)
    mean_signal_err  : float   combined uncertainty
    meas_time        : float   UTC Unix timestamp
    date             : str     compact date 'YYYYMMDD'
    target           : str     '1' or '2'
    """

    def __init__(self, hemisphere, pwm, temp, diode, batch,
                 coarse=1, fine=20, mode='default', base_path=None):

        self.diode  = diode
        self.pwm    = pwm
        self.coarse = coarse
        self.fine   = fine
        self.mode   = mode
        self.temp   = temp
        self.batch  = batch

        h5_path = base_path.format(batch=batch, hem=hemisphere)
        h = h5.File(h5_path + diode, 'r')

        # Try target '1', fall back to '2'
        for t in ('1', '2'):
            driver = _driver_key(diode, t)
            key = _data_key(diode, driver, pwm, coarse, fine, mode, temp)
            try:
                grp = h[key]
                self.vals    = np.array(grp.get('intensity_signal'))
                self.bg_vals = np.array(grp.get('intensity_bg'))
                self.target  = t
                self.driver  = driver
                break
            except KeyError:
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

        # Signal processing
        self.mean_bg_vals     = np.mean(self.bg_vals)
        self.std_bg_vals      = np.std(self.bg_vals)
        signal_vals           = self.vals - self.mean_bg_vals
        self.mean_signal_vals = -1.0 * np.mean(signal_vals)   # invert to positive
        self.std_signal_vals  = np.std(signal_vals)
        self.mean_signal_err  = np.sqrt(self.std_signal_vals**2 + self.std_bg_vals**2)


# ---------------------------------------------------------------------------
# PMT data loader
# ---------------------------------------------------------------------------

import numpy as np
import h5py as h5
import scipy.interpolate
import scipy.integrate

class SinglePMTData:
    """
    Load and pre-process one PMT waveform measurement set.

    Attributes
    ----------
    processed_data : dict  with keys:
        integrated_signal, peak, peak_time,
        trigger_rising_edge, start_integration, end_integration
    """

    def __init__(self, hemisphere, pwm, temp, target, diode, batch,
                 coarse=1, fine=20, mode='default', base_path=None, info=False):

        self.diode      = diode
        self.hemisphere = hemisphere
        self.target     = target
        self.driver     = _driver_key(diode, target)
        self.pwm        = pwm
        self.coarse     = coarse
        self.fine       = fine
        self.mode       = mode
        self.temp       = temp
        self.batch      = batch

        h5_path = base_path.format(batch=batch, hem=hemisphere)
        h = h5.File(h5_path + diode, 'r')
        key = _data_key(diode, self.driver, pwm, coarse, fine, mode, temp)
        self.pmt_time_ns = np.array(h[key].get('pmt_time_ns'))
        self.pmt_signal  = np.array(h[key].get('pmt_signal'))
        self.pmt_trigger = np.array(h[key].get('pmt_trigger'))
        h.close()

        processed = {k: [] for k in (
            'integrated_signal', 'peak', 'peak_time',
            'trigger_rising_edge', 'start_integration', 'end_integration')}

        self.data_mv_signal = [np.abs(self.pmt_signal[i])
                               for i in range(len(self.pmt_signal))]

        for i, waveform in enumerate(self.data_mv_signal):
            try:
                peak      = np.max(waveform)
                peak_idx  = np.argmax(waveform)
                peak_time = self.pmt_time_ns[peak_idx]

                interp = scipy.interpolate.InterpolatedUnivariateSpline(
                    self.pmt_time_ns, waveform)

                # Find last zero before peak
                left_zeros = [j for j, v in enumerate(waveform[:peak_idx]) if v == 0]
                if left_zeros:
                    start = self.pmt_time_ns[max(left_zeros)]
                else:
                    start = self.pmt_time_ns[0]

                # Find first zero after peak
                right_zeros = [j for j, v in enumerate(waveform[peak_idx + 1:]) if v == 0]
                if right_zeros:
                    end = self.pmt_time_ns[peak_idx + 1 + min(right_zeros)]
                else:
                    end = self.pmt_time_ns[-1]

                integrated = scipy.integrate.quad(interp, start, end)[0]

                # Find first trigger sample above threshold
                rising_edge = next(
                    (self.pmt_time_ns[j] for j in range(len(self.pmt_time_ns))
                     if self.pmt_trigger[i][j] >= 1300),
                    None)

                processed['integrated_signal'].append(integrated)
                processed['peak'].append(peak)
                processed['peak_time'].append(peak_time)
                processed['trigger_rising_edge'].append(rising_edge)
                processed['start_integration'].append(start)
                processed['end_integration'].append(end)

            except Exception as e:
                print(f'  Waveform {i} skipped: {e}')

        for k in processed:
            processed[k] = np.array(processed[k])

        self.processed_data = processed

        if info:
            self._print_info()

    def _print_info(self):
        pd = self.processed_data
        n = len(pd['integrated_signal'])
        print(f'\nPMT info ({n}/{len(self.pmt_signal)} waveforms):')
        print(f"  mean integrated signal : {np.mean(pd['integrated_signal']):.2f} mV·ns")
        print(f"  mean peak              : {np.mean(pd['peak']):.2f} mV")


# ---------------------------------------------------------------------------
# Angular calibration data
# ---------------------------------------------------------------------------

class AngularCalibration:
    """
    Load angular calibration data from a cali_flange HDF5 file.

    Provides zenith angles, current arrays, zero reference, and pulse time.
    """

    _NSIDE = 2 ** 2

    def __init__(self, file_path, batch):

        self.data   = h5.File(file_path, 'r+')
        self.result = {}
        try:
            self.hemisphere_sn = self.data['meta'].attrs['AB_SN']
        except KeyError:
            print('  WARNING: no metadata accessible')
        if batch == 'batch2':
            ipix         = hp.query_strip(self._NSIDE, np.radians(0), np.radians(150))
            deg          = np.degrees(hp.pix2ang(nside=self._NSIDE, ipix=ipix))
            self.zenith  = np.round(deg[0], 2)
            self.azimuth = np.round(deg[1], 2)
            self.zeniths = np.unique(self.zenith)
            self.indices = np.where(np.diff(self.azimuth) < 0)[0]
        if batch == 'batch1':
            self.zenith  = self.data['meta'].attrs['u_zenith']
            self.azimuth = self.data['meta'].attrs['u_azimuth']

    def pulse_time(self):
        return float(self.data['meta'].attrs['PulseTime']) * 1e-6

    def distance(self):
        return float(self.data['LMG405'].attrs['distance'])

    def curr_batch2(self, key):
        arr    = np.array(self.data.get(key))
        y      = arr[2] * 1e12
        y_err  = arr[3] * 1e12
        chunks = np.split(y,      self.indices + 1)
        cerrs  = np.split(y_err,  self.indices + 1)
        y1 = [np.mean(c) for c in chunks]
        y2 = [np.sqrt(np.std(c)**2 + np.mean(np.array(e)**2))
              for c, e in zip(chunks, cerrs)]
        return np.array([y1]), np.array(y2)

    def curr_batch1(self, key):
        arr      = np.array(self.data.get(key))
        self.y1  = arr[2] * 1e12
        self.y2  = arr[3] * 1e12
        return np.array([self.y1, self.y2])

    def zero(self, key):
        return self.data[key].attrs['zero_data'][0]

    def close(self):
        self.data.close()


# ---------------------------------------------------------------------------
# Baseline photon count
# ---------------------------------------------------------------------------

def photons_baseline(hemisphere, diode, batch, base_path):
    """
    Compute the baseline number of photons per pulse from angular calibration data.

    Returns
    -------
    photons : float
    hem     : AngularCalibration  (carries .result with error terms)
    """
    popt_air, popt_ice = fit_correction_curves()

    file_path = base_path.format(batch=batch, hem=hemisphere)
    hem = AngularCalibration(file_path + f'cali_flange_{hemisphere}', batch)

    if batch == 'batch1':
        y, y_err = hem.curr_batch1(diode)
        y_norm   = y[0][0]
        val      = np.array(y[:, 0]) / y_norm
        zenith   = hem.zenith * np.pi / 180

        values = apply_ice_correction(val, zenith, popt_ice, popt_air)
        sim_mismatch    = 0.025
        data_points_err = np.sqrt((y_err[:, 0] * 1e-12)**2 + (values * sim_mismatch)**2)

        popt, pcov = curve_fit(sigmoid, zenith, values,
                               p0=[1.1, -5.0, 1.5, -0.005], maxfev=1000)
        pcov   = pcov * 1e-12
        y_norm = y_norm * 1e-12

    if batch == 'batch2':
        y, y_err = hem.curr_batch2(diode)
        y_norm   = hem.zero(diode)
        val      = np.array(y[0]) * 1e-12 / y_norm
        zenith   = hem.zeniths * np.pi / 180

        values = apply_ice_correction(val, zenith, popt_ice, popt_air)
        sim_mismatch    = 0.025
        data_points_err = np.sqrt((y_err * 1e-12)**2 + (values * sim_mismatch)**2)

        popt, pcov = curve_fit(sigmoid, zenith, values,
                               p0=[1.1, -5.0, 1.5, -0.005], maxfev=1000)

    lb      = float(diode[-3:])
    caltrig = hem.pulse_time()
    current = np.abs(y_norm)
    d       = 96.0
    A_PD    = 1.0
    Geo     = d**2 / A_PD

    num = calc_photons(current, lb * 1e-9, NIST[diode][0], caltrig, Geo)

    N_mc          = 1000
    param_samples = np.random.multivariate_normal(popt, pcov, size=N_mc)
    photon_samples = [integrate_solid_angle(s) * 2 * np.pi * num
                      for s in param_samples]

    photons         = float(np.mean(photon_samples))
    photons_int_err = float(np.std(photon_samples))

    delta_d   = 0.5
    delta_R   = NIST[diode][1]
    R         = NIST[diode][0]
    rel_I_err = np.sqrt(np.mean((y_err * 1e-12 / y[0])**2))

    p_d_err  = 2 * delta_d / d * photons
    p_R_err  = delta_R / R      * photons
    p_I_err  = rel_I_err        * photons
    p_ps_err = 0.01             * photons
    p_sim_err = 0.025           * photons

    hem.result[diode + '_rel_sys_error']       = (abs(p_d_err) + abs(p_R_err) + p_ps_err) / photons
    hem.result[diode + '_rel_stat_error']      = np.sqrt(photons_int_err**2 + p_I_err**2 + p_sim_err**2) / photons
    hem.result[diode + '_mean_rel_sys_error']  = hem.result[diode + '_rel_sys_error']
    hem.result[diode + '_mean_rel_stat_error'] = (1 / np.sqrt(50)) * hem.result[diode + '_rel_stat_error']

    hem.close()
    return photons, hem


# ---------------------------------------------------------------------------
# Main photon number class  (PD + PMT)
# ---------------------------------------------------------------------------

class NumberOfPhotons:
    """
    Calculate the number of emitted photons per pulse for a hemisphere/emitter,
    and collect the corresponding PMT waveform summary.

    Parameters
    ----------
    hemisphere : str   e.g. '51'
    temp       : int   measurement temperature [°C]
    diode      : str   e.g. 'LMG405'
    pwm        : int   power setting
    coarse     : int   LMG coarse setting
    fine       : int   LMG fine setting
    mode       : str   KAPU mode
    batch      : str   'batch1' or 'batch2'
    base_path  : str   path template with {batch} and {hem} placeholders

    Attributes
    ----------
    final_numbers : dict
        emitted_photons_pd            – photon count from PD measurement
        emitted_photons_pd_rel_err    – single-run relative uncertainty
        emitted_photons_pd_mean_rel_err – multi-pulse mean relative uncertainty
    pmt_summary : dict
        mean_integrated_signal_mv_ns  – mean integrated waveform [mV·ns]
        mean_peak_mv                  – mean peak amplitude [mV]
        saturation_flag               – True if any waveform exceeds PMT_SATURATION_PEAK
        n_saturated_waveforms         – count of saturated waveforms
        n_total_waveforms             – count of successfully processed waveforms
        saturation_threshold_mv       – threshold used (= PMT_SATURATION_PEAK)
    meas_time : float   UTC Unix timestamp
    date      : str     compact date string 'YYYYMMDD'
    target    : str     '1' or '2'
    """

    def __init__(self, hemisphere, temp, diode, pwm, batch,
                 coarse=1, fine=20, mode='default',
                 base_path=None):

        self.diode      = diode
        self.hemisphere = hemisphere
        self.batch      = batch
        self.pwm        = pwm
        self.coarse     = coarse
        self.fine       = fine
        self.mode       = mode
        self.temp       = temp

        # ------------------------------------------------------------------ #
        # PD-based photon number
        # ------------------------------------------------------------------ #

        # Measurement PD data
        data_pd_obj = SinglePDData(
            hemisphere=hemisphere, pwm=pwm, temp=temp,
            diode=diode, coarse=coarse, fine=fine, mode=mode,
            batch=batch, base_path=base_path)
        data_pd         = data_pd_obj.mean_signal_vals
        data_pd_rel_err = data_pd_obj.mean_signal_err / data_pd
        self.meas_time  = data_pd_obj.meas_time
        self.date       = data_pd_obj.date
        self.target     = data_pd_obj.target

        # Baseline (reference) PD data  — always pwm=54000, temp=25, default settings
        norm_pd_obj = SinglePDData(
            hemisphere=hemisphere, pwm=54000, temp=25,
            diode=diode, coarse=1, fine=20, mode='default',
            batch=batch, base_path=base_path)
        norm_pd         = norm_pd_obj.mean_signal_vals
        norm_pd_rel_err = norm_pd_obj.mean_signal_err / norm_pd

        # Baseline photon count from angular calibration
        baseline, hem = photons_baseline(
            hemisphere=hemisphere, diode=diode,
            batch=batch, base_path=base_path)

        emitted_pd = data_pd / norm_pd * baseline
        if emitted_pd < 0:
            print('  WARNING: negative photon count → set to 0')
            emitted_pd = 0.0

        b_sys  = hem.result[diode + '_rel_sys_error']
        b_stat = hem.result[diode + '_rel_stat_error']
        b_msys = hem.result[diode + '_mean_rel_sys_error']
        b_mst  = hem.result[diode + '_mean_rel_stat_error']

        rel_err = np.sqrt(data_pd_rel_err**2 + norm_pd_rel_err**2
                          + b_stat**2 + b_sys**2)
        mean_rel_err = np.sqrt((data_pd_rel_err / np.sqrt(100))**2
                               + (norm_pd_rel_err / np.sqrt(100))**2
                               + b_mst**2 + b_msys**2)

        self.final_numbers = {
            'emitted_photons_pd':              emitted_pd,
            'emitted_photons_pd_rel_err':      rel_err,
            'emitted_photons_pd_mean_rel_err': mean_rel_err,
        }

        # ------------------------------------------------------------------ #
        # PMT waveform summary
        # ------------------------------------------------------------------ #
        
        data_pmt_obj = SinglePMTData(
            hemisphere=hemisphere, pwm=pwm, temp=temp,
            target=self.target,           # use the same target resolved by PD
            diode=diode, coarse=coarse, fine=fine, mode=mode,
            batch=batch, base_path=base_path)
        
        norm_pmt_obj = SinglePMTData(
            hemisphere=hemisphere, pwm=54000, temp=25,target=self.target,
            diode=diode, coarse=1, fine=20, mode='default',
            batch=batch, base_path=base_path)

        signals = data_pmt_obj.processed_data['integrated_signal']
        norm_signals = norm_pmt_obj.processed_data['integrated_signal']  # FIXED LINE

        for norm_pwm in [54000, 45000, 35000, 30000, 25000, 20000, 15000, 10000, 7500]: 

            norm_pmt_obj_dict = SinglePMTData(hemisphere=hemisphere, pwm=norm_pwm, temp=temp, target=self.target,
                                        coarse=1, fine=20, mode='default', diode=diode, batch=batch, base_path=base_path)
            norm_pmt_dict = norm_pmt_obj_dict.processed_data

            if np.max(norm_pmt_dict['peak'] ) < 4500:
                break
            else:
                pass

        norm_pmt = np.mean( norm_pmt_dict['integrated_signal'] )

        norm_cross = SinglePDData(hemisphere=hemisphere, pwm=norm_pwm, temp=temp,
                                    coarse=coarse, fine=fine, mode=mode, diode=diode, batch=batch, base_path=base_path).mean_signal_vals

        emitted_pmt = np.mean(signals) / np.mean(norm_signals) * (norm_cross / norm_pd)* baseline

        peaks = data_pmt_obj.processed_data['peak']
        n_saturated = int(np.sum(peaks >= PMT_SATURATION_PEAK))
        n_total = len(peaks)
        self.pmt_summary = {
            'n_photons': float(emitted_pmt),
            'mean_integrated_signal_mv_ns': float(np.mean(signals)),
            'mean_peak_mv': float(np.mean(peaks)),
            'saturation_flag': n_saturated > 0,
            'n_saturated_waveforms': n_saturated,
            'n_total_waveforms': n_total,
            'saturation_threshold_mv': PMT_SATURATION_PEAK,
            }

# ---------------------------------------------------------------------------
# JSON assembly and file writing
# ---------------------------------------------------------------------------

def compute_and_save(hemisphere, device_id, emitter,
                     pwm, temp, coarse, fine, mode, target, batch,
                     paths):
    """
    Compute photon number (PD + PMT) and write a single JSON output file.

    Parameters
    ----------
    hemisphere : str   HDF5 hemisphere id
    device_id  : str   POCAM device number e.g. '016'
    emitter    : str   e.g. 'LMG405'
    pwm        : int   power setting
    temp       : int   measurement temperature [°C]
    coarse     : int   LMG coarse setting (ignored for KAPU)
    fine       : int   LMG fine setting   (ignored for KAPU)
    mode       : str   KAPU mode          (ignored for LMG)
    target     : str   '1' or '2' — passed to NumberOfPhotons for PMT loading
    batch      : str   'batch1' or 'batch2'
    paths      : dict  with keys 'data' and 'output'

    Returns
    -------
    dict — assembled result dictionary (also written to disk as JSON)
    """
    base_path = paths['data']
    driver    = _driver_char(emitter)

    # Single object now owns both PD and PMT results
    photons = NumberOfPhotons(
        hemisphere=hemisphere, temp=temp, diode=emitter, pwm=pwm,
        coarse=coarse, fine=fine, mode=mode,
        batch=batch, base_path=base_path)

    target_label = 'master' if photons.target == '1' else 'slave'

    # ------------------------------------------------------------------ #
    # meas_data entry 1 — PD-based photon count
    # ------------------------------------------------------------------ #
    pd_entry = {
        'data_format': 'value',
        'value':       round(photons.final_numbers['emitted_photons_pd'], 2),
        'error':       round(photons.final_numbers['emitted_photons_pd_rel_err'], 4),
        'power':       photons.pwm,
        'temperature': photons.temp,
        'label':       'Number-of-Emitted-Photons-per-Pulse',
    }
    if 'LMG' in emitter:
        pd_entry['coarse'] = photons.coarse
        pd_entry['fine']   = photons.fine
    else:
        pd_entry['mode'] = photons.mode

    mean_rel_err = round(photons.final_numbers['emitted_photons_pd_mean_rel_err'], 3)

    # ------------------------------------------------------------------ #
    # meas_data entry 2 — PMT waveform summary
    # ------------------------------------------------------------------ #
    ps = photons.pmt_summary
    pmt_entry = {
        'data_format':                  'summary',
        'label':                        'PMT-Waveform-Summary',
        'power':                        photons.pwm,
        'temperature':                  photons.temp,
        'value': ps['n_photons'],
        'mean_integrated_signal_mv_ns': round(ps['mean_integrated_signal_mv_ns'], 4),
        # 'mean_peak_mv':                 round(ps['mean_peak_mv'], 4),
        # 'saturation_flag':              ps['saturation_flag'],
        # 'n_saturated_waveforms':        ps['n_saturated_waveforms'],
        # 'n_total_waveforms':            ps['n_total_waveforms'],
        # 'saturation_threshold_mv':      ps['saturation_threshold_mv'],
    }
    if 'LMG' in emitter:
        pmt_entry['coarse'] = photons.coarse
        pmt_entry['fine']   = photons.fine
    else:
        pmt_entry['mode'] = photons.mode

    # ------------------------------------------------------------------ #
    # Assemble full result
    # ------------------------------------------------------------------ #
    result = {
        'device_uid':    f'pocam-{photons.date}_{device_id}',
        'subdevice_uid': f'pocam-led-{target_label}_{driver}-{emitter[-3:]}_{device_id}',
        'meas_name':     'led-number-of-emitted-photons',
        'meas_class':    'display',
        'meas_stage':    'calibration',
        'meas_group':    'luminosity',
        'meas_site':     'tum',
        'meas_time':     photons.meas_time,
        'meas_data':     [pd_entry, pmt_entry],
        # 'comments': [
        #     "Total emitted photons per pulse for this hemisphere over the full solid angle "
        #     "at the given conditions.",
        #     "Value based on repeated PD light-output measurements.",
        #     f"Mean relative error for multi-pulse runs: {mean_rel_err}",
        #     f"PMT saturation flag: {ps['saturation_flag']} "
        #     f"({ps['n_saturated_waveforms']}/{ps['n_total_waveforms']} waveforms "
        #     f"above {PMT_SATURATION_PEAK} mV threshold).",
        # ],
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

    if 'LMG' in emitter:
        fname = f'photons_{pwm}_{temp}C_{coarse}-{fine}.json'
    else:
        fname = f'photons_{pwm}_{temp}C_{mode}.json'

    out_path = os.path.join(paths['output'], f'{batch}',
                            f'pocam_{device_id}',
                            f'hem_{hemisphere}',
                            f'{emitter}',
                            fname)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f'  Saved → {out_path}')

    return result