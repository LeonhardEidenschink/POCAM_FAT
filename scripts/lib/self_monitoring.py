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
    format_date,
    datetime_to_unix
)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

HC = 1.98644586e-25          # h·c  [m³ kg / s²]
A_PD = 1.0                   # photodiode active area [cm²]
DIST_CM = 96.0               # flange-equator to PD surface [cm]
PMT_TRIGGER_THRESHOLD = 1300 # trigger rising-edge threshold [mV]

# Air-to-ice transmission correction look-up table (from calibration)
_X_PRE = np.array(X_PRE)
_Y_PRE = np.array(Y_PRE)
_X_VALS = np.array(X_VALUES)
_Y_VALS = np.array(Y_VALUES)


# ---------------------------------------------------------------------------
# SinglePDData
# ---------------------------------------------------------------------------

class SinglePDData:
    """
    Load and pre-process one PD picoamp measurement set.

    Parameters
    ----------
    hemisphere : str    e.g. '51'
    pwm        : int    power setting
    temp       : int    temperature [°C]
    diode      : str    e.g. 'LMG405'
    target     : str    '1' or '2'
    coarse     : int    LMG coarse setting
    fine       : int    LMG fine setting
    mode       : str    KAPU mode ('default' or 'fast')
    base_path  : str    path template with {hem} placeholder

    Attributes
    ----------
    mean_signal_vals : float  background-subtracted mean signal [V]
    mean_signal_err  : float  combined statistical uncertainty
    """

    def __init__(self, hemisphere, pwm, temp, diode,batch,
                 target='1', coarse=1, fine=20, mode='default',
                 base_path=None):

        self.diode     = diode
        self.target    = target
        self.driver    = ('lmg' if 'LMG' in diode else 'kapu') + target
        self.pwm       = pwm
        self.coarse    = coarse
        self.fine      = fine
        self.mode      = mode
        self.temp      = temp
        self.batch     = batch

        h5_path = base_path.format(batch = batch, hem=hemisphere)
        h = h5.File(h5_path + diode, 'r')
        key = _build_key(diode, self.driver, pwm, coarse, fine, mode, temp)

        self.vals    = np.array(h[key].get('intensity_signal'))
        self.bg_vals = np.array(h[key].get('intensity_bg'))
        h.close()

        self.mean_bg_vals  = np.mean(self.bg_vals)
        self.std_bg_vals   = np.std(self.bg_vals)

        signal_vals            = self.vals - self.mean_bg_vals
        self.mean_signal_vals  = -1.0 * np.mean(signal_vals)   # invert to positive
        self.std_signal_vals   = np.std(signal_vals)
        self.mean_signal_err   = np.sqrt(self.std_signal_vals**2
                                         + self.std_bg_vals**2)


# ---------------------------------------------------------------------------
# SinglePMTData
# ---------------------------------------------------------------------------

class SinglePMTData:
    """
    Load and pre-process one PMT waveform dataset.

    Parameters
    ----------
    Same signature as SinglePDData (hemisphere, pwm, temp, diode, …).

    Attributes
    ----------
    processed_data : dict with keys
        integrated_signal, peak, peak_time,
        trigger_rising_edge, start_integration, end_integration
    """

    def __init__(self, hemisphere, pwm, temp, diode, batch, 
                 target='1', coarse=1, fine=20, mode='default',
                 base_path=None, info=False):

        self.diode  = diode
        self.target = target
        self.driver = ('lmg' if 'LMG' in diode else 'kapu') + target
        self.pwm    = pwm
        self.coarse = coarse
        self.fine   = fine
        self.mode   = mode
        self.temp   = temp
        self.batch  = batch

        h5_path = base_path.format(batch = batch, hem=hemisphere)
        h = h5.File(h5_path + diode, 'r')
        key = _build_key(diode, self.driver, pwm, coarse, fine, mode, temp)

        self.pmt_time_ns = np.array(h[key].get('pmt_time_ns'))
        self.pmt_signal  = np.array(h[key].get('pmt_signal'))
        self.pmt_trigger = np.array(h[key].get('pmt_trigger'))
        h.close()

        processed = {k: [] for k in ('integrated_signal', 'peak', 'peak_time',
                                     'trigger_rising_edge', 'start_integration',
                                     'end_integration')}

        abs_signals = [np.abs(wf) for wf in self.pmt_signal]

        for i, sig in enumerate(abs_signals):
            try:
                peak       = np.max(sig)
                peak_idx   = np.argmax(sig)
                peak_time  = self.pmt_time_ns[peak_idx]
                interp     = scipy.interpolate.InterpolatedUnivariateSpline(
                                 self.pmt_time_ns, sig)

                left_times  = self.pmt_time_ns[:peak_idx]
                left_zeros  = np.where(sig[:peak_idx] == 0)[0]
                start       = self.pmt_time_ns[np.max(left_zeros)]

                right_times = self.pmt_time_ns[peak_idx + 1:]
                right_zeros = np.where(sig[peak_idx + 1:] == 0)[0]
                end         = right_times[np.min(right_zeros)]

                integrated  = scipy.integrate.quad(interp, start, end, limit=200)[0]

                # Trigger rising edge: first sample >= threshold
                j = 0
                while j < len(self.pmt_time_ns):
                    if self.pmt_trigger[i][j] >= PMT_TRIGGER_THRESHOLD:
                        rising_edge = self.pmt_time_ns[j]
                        break
                    j += 1

                processed['integrated_signal'].append(integrated)
                processed['peak'].append(peak)
                processed['peak_time'].append(peak_time)
                processed['trigger_rising_edge'].append(rising_edge)
                processed['start_integration'].append(start)
                processed['end_integration'].append(end)

            except Exception as e:
                print(f'  Warning: waveform {i} skipped — {e}')

        self.processed_data = {k: np.array(v) for k, v in processed.items()}

        if info:
            self._print_info()

    def _print_info(self):
        pd = self.processed_data
        n  = len(pd['integrated_signal'])
        print(f'  Processed waveforms : {n} / {len(self.pmt_signal)}')
        print(f'  Mean integrated sig : '
              f'{np.mean(pd["integrated_signal"]):.2f} mV·ns')
        print(f'  Mean peak           : {np.mean(pd["peak"]):.2f} mV')


# ---------------------------------------------------------------------------
# SingleAngularCalData
# ---------------------------------------------------------------------------

class SingleAngularCalData:
    """
    Load flange-calibration data for one hemisphere / emitter and compute the
    baseline photon number via angular integration.

    Parameters
    ----------
    hemisphere : str
    diode      : str    e.g. 'LMG405'
    base_path  : str    path template with {hem} placeholder
    cal_prefix : str    calibration file prefix (default 'cali_flange_')

    Attributes
    ----------
    photons          : float  total emitted photons per pulse
    photons_sys_err  : float  systematic error
    photons_stat_err : float  statistical error
    rel_sys_error    : float
    rel_stat_error   : float
    popt             : array  sigmoid fit parameters
    hemisphere_sn    : str    serial number from metadata
    pulse_time_s     : float  pulse duration [s]
    zero_current_A   : float  normalisation current [A]
    """

    def __init__(self, hemisphere, diode, base_path, batch, cal_prefix='cali_flange_'):

        self.diode     = diode
        self.hemisphere = hemisphere
        self.batch     = batch

        h5_path  = base_path.format(batch = batch, hem=hemisphere)
        cal_file = h5_path + cal_prefix + hemisphere
        h        = h5.File(cal_file, 'r+')

        self.hemisphere_sn = str(int(h['meta'].attrs['AB_SN']))
        self.pulse_time_s  = float(h['meta'].attrs['PulseTime']) * 1e-6
        ncal               = h['meta'].attrs['ncal']

        # ---- Angular binning (HEALPix) ----
        NSIDE    = 4
        ipix     = hp.query_strip(NSIDE, np.radians(0), np.radians(150))
        deg      = np.degrees(hp.pix2ang(nside=NSIDE, ipix=ipix))
        zenith   = np.round(deg[0], 2)
        azimuth  = np.round(deg[1], 2)
        indices  = np.where(np.diff(azimuth) < 0)[0]
        zeniths  = np.unique(zenith)

        # ---- Current readings ----
        arr        = np.array(h.get(diode))
        y_raw      = arr[2] * 1e12          # [pA]
        y_err_raw  = arr[3] * 1e12

        # Split by azimuth ring boundaries and average per zenith ring
        y_split     = np.split(y_raw,    indices + 1)
        yerr_split  = np.split(y_err_raw, indices + 1)

        y_mean = np.array([np.mean(seg) for seg in y_split])
        y_err  = np.array([
            np.sqrt(np.std(seg)**2
                    + np.sum(np.array(e)**2) / len(e))
            for seg, e in zip(y_split, yerr_split)
        ])

        self.zero_current_A = np.abs(h[diode].attrs['zero_data'][0])
        h.close()

        # ---- Air-to-ice correction ----
        popt_air, popt_ice = fit_correction_curves()
        zen_rad  = zeniths * np.pi / 180.0
        val_norm = y_mean * 1e-12 / self.zero_current_A   # normalise

        # Relative error from simulation mismatch
        sim_mismatch = 0.025
        dp_err = np.sqrt((y_err * 1e-12)**2
                         + (val_norm * sim_mismatch)**2)

        values = val_norm * (sigmoid(zen_rad, *popt_ice)
                             / sigmoid(zen_rad, *popt_air))

        # ---- Angular fit ----
        popt, pcov = curve_fit(sigmoid, zen_rad, values,
                               p0=[1.1, -5.0, 1.5, -0.005], maxfev=1000)
        self.popt = popt
        self.pcov = pcov

        # ---- Photon number (MC) ----
        resp       = NIST[diode][0]
        resp_err   = NIST[diode][1]
        wl_m       = float(diode[-3:]) * 1e-9
        geo        = DIST_CM**2 / A_PD

        baseline   = calc_photons(self.zero_current_A, wl_m, resp,
                                   self.pulse_time_s, geo)

        N_mc       = 1000
        param_samp = multivariate_normal(popt, pcov, size=N_mc)
        ph_samp    = [integrate_solid_angle(s) * 2 * np.pi * baseline
                      for s in param_samp]

        self.photons           = float(np.mean(ph_samp))
        self.photons_int_err   = float(np.std(ph_samp))

        # ---- Error budget ----
        photons_d_err   = 2 * 0.5 / DIST_CM * self.photons       # 0.5 cm distance err
        photons_R_err   = resp_err / resp * self.photons
        rel_I           = np.sqrt(np.mean((y_err * 1e-12 / y_mean * 1e-12)**2))
        photons_I_err   = rel_I * self.photons
        photons_ps_err  = 0.01 * self.photons                     # point-source approx
        photons_sim_err = 0.025 * self.photons

        self.photons_sys_err  = (np.abs(photons_d_err)
                                 + np.abs(photons_R_err)
                                 + photons_ps_err)
        self.photons_stat_err = np.sqrt(self.photons_int_err**2
                                        + photons_I_err**2
                                        + photons_sim_err**2)

        self.rel_sys_error  = self.photons_sys_err  / self.photons
        self.rel_stat_error = self.photons_stat_err / self.photons

        # Mean-measurement error (scaled by sqrt(50) for repeated measurements)
        self.mean_rel_sys_error  = self.rel_sys_error
        self.mean_rel_stat_error = self.photons_stat_err / np.sqrt(50) / self.photons


# ---------------------------------------------------------------------------
# compute_and_save
# ---------------------------------------------------------------------------
def compute_self_monitoring(hemisphere, device_id, emitter,
                             pwm, temp, coarse, fine, mode,
                             paths, batch, target=None):
    """
    Compute self-monitoring photon number for one (hemisphere, emitter, temp)
    combination. Returns (meas_data_entries, meta) — no file I/O.
    """
    base_path  = paths['data']
    cal_prefix = paths.get('cal_prefix', 'cali_flange_')

    # ---- Auto-detect target ----
    if target is None:
        h5_path = base_path.format(batch=batch, hem=hemisphere)
        h = h5.File(h5_path + emitter, 'r')
        target, _ = _resolve_target(h, emitter, pwm, coarse, fine, mode, temp)
        h.close()

    target_label = 'master' if target == '1' else 'slave'

    # ---- PD signal at requested conditions ----
    pd_data = SinglePDData(
        hemisphere=hemisphere, pwm=pwm, temp=temp, diode=emitter,
        target=target, coarse=coarse, fine=fine, mode=mode,
        batch=batch, base_path=base_path)

    # ---- PD signal at baseline (25°C, max power) ----
    pd_norm = SinglePDData(
        hemisphere=hemisphere, pwm=54000, temp=25, diode=emitter,
        target=target, coarse=1, fine=20, mode='default',
        batch=batch, base_path=base_path)

    # ---- Angular calibration (only needed once per emitter, but kept here
    #      for simplicity — caller can cache if performance matters) ----
    cal = SingleAngularCalData(
        hemisphere=hemisphere, diode=emitter, batch=batch,
        base_path=base_path, cal_prefix=cal_prefix)

    # ---- Extract date and meas_time from HDF5 metadata ----
    h5_path = base_path.format(batch=batch, hem=hemisphere)
    h = h5.File(h5_path + emitter, 'r')
    if 'LMG' in emitter:
        meta_key = f'{pd_data.driver}/{54000}/1-20/25C/metadata'
    else:
        meta_key = f'{pd_data.driver}/{54000}/default/25C/metadata'
    dt_str = h[meta_key].attrs.get('datetime')
    h.close()
    date      = format_date(dt_str)
    meas_time = datetime_to_unix(dt_str)

    # ---- Scale photons from PD ratio ----
    pd_ratio           = pd_data.mean_signal_vals / pd_norm.mean_signal_vals
    emitted_photons_pd = max(0.0, pd_ratio * cal.photons)

    pd_rel_err   = pd_data.mean_signal_err / pd_data.mean_signal_vals
    norm_rel_err = pd_norm.mean_signal_err / pd_norm.mean_signal_vals

    pd_rel_total = np.sqrt(pd_rel_err**2
                           + norm_rel_err**2
                           + cal.rel_stat_error**2
                           + cal.rel_sys_error**2)

    # ---- Build PD entry ----
    pd_entry = {
        'data_format': 'value', 
        'value':       round(emitted_photons_pd),
        'error':       round(emitted_photons_pd * pd_rel_total),
        'rel_error':   round(pd_rel_total, 4),
        'power':       pwm,
        'temperature': temp,
        'label':       'N_photons_PD',
        'title':       'Estimated Number of Emitted Photons per Pulse (PD)',
    }
    if 'LMG' in emitter:
        pd_entry['coarse'] = coarse
        pd_entry['fine']   = fine
    else:
        pd_entry['mode'] = mode

    entries = [pd_entry]

    # ---- PMT cross-calibration (best-effort) ----
    try:
        pmt_data = SinglePMTData(
            hemisphere=hemisphere, pwm=pwm, temp=temp, diode=emitter,
            target=target, coarse=coarse, fine=fine, mode=mode,
            batch=batch, base_path=base_path)
        data_pmt_mean = float(np.mean(
            pmt_data.processed_data['integrated_signal']))

        # Find highest PMT power below saturation
        norm_pmt_obj = None
        for try_pwm in [54000, 45000, 35000, 30000, 25000,
                        20000, 15000, 10000, 7500]:
            candidate = SinglePMTData(
                hemisphere=hemisphere, pwm=try_pwm, temp=temp,
                diode=emitter, target=target, coarse=1, fine=20,
                mode='default', batch=batch, base_path=base_path)
            if np.max(candidate.processed_data['peak']) < 4500:
                norm_pmt_obj = candidate
                norm_pwm     = try_pwm
                break

        if norm_pmt_obj is not None:
            norm_pmt_mean = float(np.mean(
                norm_pmt_obj.processed_data['integrated_signal']))

            # ---- Fixed: use baseline coarse/fine/mode for norm PD ----
            norm_pd_cross = SinglePDData(
                hemisphere=hemisphere, pwm=norm_pwm, temp=temp,
                diode=emitter, target=target,
                coarse=1, fine=20, mode='default',   # baseline settings
                batch=batch, base_path=base_path).mean_signal_vals

            emitted_photons_pmt = ((data_pmt_mean / norm_pmt_mean)
                                   * (norm_pd_cross / pd_norm.mean_signal_vals)
                                   * cal.photons)

            pmt_entry = {
                'data_format': 'value',
                'value':       round(emitted_photons_pmt),
                'power':       pwm,
                'temperature': temp,
                'label':       'N_photons_PMT',
                'title':       'Estimated Number of Emitted Photons per Pulse (PMT cross-cal)',
            }
            if 'LMG' in emitter:
                pmt_entry['coarse'] = coarse
                pmt_entry['fine']   = fine
            else:
                pmt_entry['mode'] = mode
            entries.append(pmt_entry)

    except Exception as e:
        print(f'    PMT cross-calibration skipped — {e}')

    # ---- Metadata (used once per emitter to build top-level JSON fields) ----
    meta = {
        'date':          date,
        'meas_time':     meas_time,
        'target_label':  target_label,
        'hemisphere_sn': cal.hemisphere_sn,
        'pulse_time_us': cal.pulse_time_s * 1e6,
        'driver':        pd_data.driver,
    }

    return entries, meta

def save_emitter_json(hemisphere, device_id, emitter,
                      batch, meas_data_list, meta, paths):
    """
    Write one JSON file containing all meas_data entries for this
    hemisphere/emitter combination (all temperatures).
    """
    driver_char  = 'l' if 'LMG' in emitter else 'k'
    target_label = meta['target_label']

    result = {
        'device_uid':    f'pocam-{meta["date"]}_{device_id}',
        'subdevice_uid': (f'pocam-led-{target_label}_{driver_char}'
                          f'-{emitter[-3:]}_{device_id}'),
        'meas_name':     'intensity-photon-number',
        'meas_class':    'display',
        'meas_stage':    'calibration',
        'meas_group':    'luminosity',        # fixed from 'intensity'
        'meas_site':     'tum',
        'meas_time':     meta['meas_time'],   # now present
        'meas_data':     meas_data_list,      # all temperatures
        'comments': [
            f'Estimated total number of emitted photons per pulse for hemisphere {hemisphere} in POCAM {device_id}.',
            'Method: PD picoamp signal ratio × angular-integration baseline.',
            'Baseline derived from flange-calibration HDF5 data.',
            'Air-to-ice correction applied via pre-measured sigmoid model.',
            f'Hemisphere SN : {meta["hemisphere_sn"]}',
            f'Pulse duration: {meta["pulse_time_us"]:.3f} µs',
        ],
        'support_files': [
            {
                'filetype': 'hdf5',
                'hostname': 'data.icecube.wisc.edu',
                'pathname': (f'/data/exp/IceCubeUpgrade/commissioning/pocam/'
                             f'pocam_{device_id}/{target_label}_hemisphere/{emitter}'),
                'comment':  'Raw HDF5 data.',
            },
            {
                'filetype': 'pdf',
                'hostname': 'data.icecube.wisc.edu',
                'pathname': ('/data/exp/IceCubeUpgrade/commissioning/'
                             'pocam/POCAM_documentation.pdf'),
                'comment':  'POCAM documentation guide.',
            },
        ],
    }

    out_path = os.path.join(paths['output'], batch,
                            f'pocam_{device_id}',
                            f'hem_{hemisphere}',
                            emitter,
                            'self_monitoring.json')   # one file per emitter
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f'  Saved → {out_path}')
    return result