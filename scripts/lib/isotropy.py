"""
isotropy.py
===============
Library of isotropy classes and JSON-building helpers for POCAM devices.
Import this from your run script — do not run directly.
"""

import json
import numpy as np
import h5py
import healpy as hp
import os

from pocam_utils import (
    fit_correction_curves,
    apply_ice_correction,
    date_to_unix,
    compute_isotropy,
)


# ---------------------------------------------------------------------------
# Data paths  (passed in from config — not hardcoded here)
# ---------------------------------------------------------------------------

def _hem_path(batch, hem, base_paths):
    return base_paths[batch].format(hem=hem)


def _driver(emitter):
    return 'l' if 'LMG' in emitter else 'k'


# ---------------------------------------------------------------------------
# Batch 1 — meshgrid angular scan
# ---------------------------------------------------------------------------

class IsotropyBatch1:
    """
    Compute isotropy for a batch1 POCAM device.

    Parameters
    ----------
    hemispheres : list of str   e.g. ['11', '12']
    diode       : str           e.g. 'LMG405'

    Attributes
    ----------
    y_values        : list   normalised angular emission profile
    isotropy_value  : float
    isotropy_error  : float
    date1, date2    : str    measurement date strings
    meas_time       : float  UTC Unix timestamp
    """

    def __init__(self, hemispheres=('11', '12'), diode='LMG405', base_paths=None):
        self.hemispheres = list(hemispheres)
        self.diode       = diode

        popt_air, popt_ice = fit_correction_curves()

        path1 = _hem_path('batch1', hemispheres[0], base_paths)
        path2 = _hem_path('batch1', hemispheres[1], base_paths)

        with h5py.File(path1, 'r') as h1, h5py.File(path2, 'r') as h2:
            self.date1     = h1['meta'].attrs['date']
            self.date2     = h2['meta'].attrs['date']
            self.meas_time = date_to_unix(self.date1)

            data_1 = np.array(h1.get(diode))   # [0] = mean, [1] = error
            data_2 = np.array(h2.get(diode))
            print(len(h1['meta'].attrs['zenith']))
            print(h1['meta'].attrs['zenith'])
            zen    = h1['meta'].attrs['zenith'].reshape(16, 6)

        data1, err1 = data_1[0], data_1[1]
        data2, err2 = data_2[0], data_2[1]

        # Normalise each hemisphere to its minimum
        mini1, mini2 = data1.min(), data2.min()
        y1,   y2     = data1 / mini1,  data2 / mini2
        err1, err2   = err1  / mini1,  err2  / mini2

        # Ice correction
        y1   = apply_ice_correction(y1,   zen, popt_ice, popt_air)
        y2   = apply_ice_correction(y2,   zen, popt_ice, popt_air)
        err1 = apply_ice_correction(err1, zen, popt_ice, popt_air)
        err2 = apply_ice_correction(err2, zen, popt_ice, popt_air)

        # Pad rows for south-pole region
        pad  = np.zeros((3, 6))
        y1   = np.concatenate([y1,   pad], axis=0)
        y2   = np.concatenate([y2,   pad], axis=0)
        err1 = np.concatenate([err1, pad], axis=0)
        err2 = np.concatenate([err2, pad], axis=0)

        y_total     = y1   + np.flip(y2)
        y_total_err = err1 + np.flip(err2)

        self.y_values, _, self.isotropy_value, self.isotropy_error = \
            compute_isotropy(y_total, y_total_err)
        self.y_values = self.y_values.tolist()


# ---------------------------------------------------------------------------
# Batch 2 — HEALPix angular scan
# ---------------------------------------------------------------------------

class IsotropyBatch2:
    """
    Compute isotropy for a batch2 POCAM device.

    Parameters
    ----------
    hemispheres : list of str   e.g. ['35', '36']
    diode       : str           e.g. 'LMG405'

    Attributes
    ----------
    y_values        : list   normalised angular emission profile
    isotropy_value  : float
    isotropy_error  : float
    date1, date2    : str    measurement date strings
    meas_time       : float  UTC Unix timestamp
    """

    _NSIDE = 2 ** 2

    def __init__(self, hemispheres=('35', '36'), diode='LMG405', base_paths=None):
        self.hemispheres = list(hemispheres)
        self.diode       = diode

        ipix = hp.query_strip(self._NSIDE, np.radians(0), np.radians(150))
        deg  = np.degrees(hp.pix2ang(nside=self._NSIDE, ipix=ipix))

        popt_air, popt_ice = fit_correction_curves()

        path1 = _hem_path('batch2', hemispheres[0], base_paths)
        path2 = _hem_path('batch2', hemispheres[1], base_paths)

        with h5py.File(path1, 'r') as h1, h5py.File(path2, 'r') as h2:
            self.date1     = h1['meta'].attrs['date']
            self.date2     = h2['meta'].attrs['date']
            self.meas_time = date_to_unix(self.date1)

            # batch2 layout: index 2 = mean, index 3 = error
            data1 = np.array(h1.get(diode)[2])
            data2 = np.array(h2.get(diode)[2])
            err1  = np.array(h1.get(diode)[3])
            err2  = np.array(h2.get(diode)[3])

        y1,  y2  = data1 / data1.min(), data2 / data2.min()
        ye1, ye2 = err1  / data1.min(), err2  / data2.min()

        y1   = apply_ice_correction(y1,  deg[0], popt_ice, popt_air)
        y2   = apply_ice_correction(y2,  deg[0], popt_ice, popt_air)
        ye1  = apply_ice_correction(ye1, deg[0], popt_ice, popt_air)
        ye2  = apply_ice_correction(ye2, deg[0], popt_ice, popt_air)

        zeros = np.zeros(12)
        y1,  y2  = np.concatenate([y1,  zeros]), np.concatenate([y2,  zeros])
        ye1, ye2 = np.concatenate([ye1, zeros]), np.concatenate([ye2, zeros])

        y_total     = y1  + np.flip(y2)
        y_total_err = ye1 + np.flip(ye2)

        self.y_values, _, self.isotropy_value, self.isotropy_error = \
            compute_isotropy(y_total, y_total_err)
        self.y_values = self.y_values.tolist()


# ---------------------------------------------------------------------------
# JSON assembly and file writing
# ---------------------------------------------------------------------------

def _build_grid_values(iso, batch, coarse, fine, mode, pwm, temp):
    gv = {
        'data_format': 'healpix' if batch == 'batch2' else 'meshgrid',
        'projection':  'mollview',
        'power':        pwm,
        'temperature':  temp,
        'x_label':     'azimuth angle phi [°]',
        'y_label':     'zenith angle theta [°]',
        'z_label':     'relative emission intensity',
        'x_min':       -np.pi,  'x_max': np.pi,
        'y_min':       -np.pi / 2, 'y_max': np.pi / 2,
        'z_values':     iso.y_values,
        'title':        'angular-emission-profile',
    }
    if 'LMG' in iso.diode:
        gv['coarse'] = coarse
        gv['fine']   = fine
    else:
        gv['mode'] = mode

    if batch == 'batch2':
        gv['bins'] = len(iso.y_values)
    else:
        gv['x_bins'] = 6
        gv['y_bins'] = 19

    return gv


def _build_value_entry(iso, coarse, fine, mode, pwm, temp):
    ve = {
        'data_format': 'value',
        'value':        round(iso.isotropy_value, 3),
        'error':        round(iso.isotropy_error, 3),
        'power':        pwm,
        'temperature':  temp,
        'label':        'Level-of-Isotropy-for-entire-Device',
    }
    if 'LMG' in iso.diode:
        ve['coarse'] = coarse
        ve['fine']   = fine
    else:
        ve['mode'] = mode
    return ve


def compute_and_save(device_id, hemispheres, batch, emitter,
                     coarse, fine, mode, pwm, temp, target, paths):
    """
    Core routine: compute isotropy and write JSON output files.

    Called by run_isotropy.py for each (device, emitter) combination.

    Returns
    -------
    dict — assembled isotropy result
    """
    driver = _driver(emitter)

    if batch == 'batch1':
        iso = IsotropyBatch1(hemispheres=hemispheres, diode=emitter, base_paths={'batch1':paths['batch1_data'], 'batch2': paths['batch2_data']})
    elif batch == 'batch2':
        iso = IsotropyBatch2(hemispheres=hemispheres, diode=emitter, base_paths={'batch1':paths['batch1_data'], 'batch2': paths['batch2_data']})
    else:
        raise ValueError(f"Unknown batch '{batch}' for device {device_id}")

    isotropy = {
        'device_uid':    f'pocam-{iso.date1}_{device_id}',
        'subdevice_uid': f'pocam-led-master_{driver}-{emitter[-3:]}_{device_id}',
        'meas_name':     'led-isotropy-level',
        'meas_class':    'display',
        'meas_stage':    'calibration',
        'meas_group':    'isotropy',
        'meas_site':     'tum',
        'meas_time':     iso.meas_time,
        'meas_batch':    batch,
        'meas_data': [
            _build_grid_values(iso, batch, coarse, fine, mode, pwm, temp),
            _build_value_entry(iso, coarse, fine, mode, pwm, temp),
        ],
        'comments': [
            "Isotropy in [%]: half the peak-to-peak range of the normalised angular "
            "emission profile, obtained by combining both hemispheres.",
            f"Hemispheres {hemispheres[0]} and {hemispheres[1]} installed in POCAM {device_id}.",
            "phi: rotation around long POCAM axis. theta: perpendicular to phi.",
            "Azimuthal range 0–360°; zenith range 0–150°.",
        ],
        'support_files': [
            {
                'filetype': 'hdf5',
                'hostname': 'data.icecube.wisc.edu',
                'pathname': f'/data/exp/IceCubeUpgrade/commissioning/pocam/'
                            f'pocam_{device_id}/{target}_hemisphere/{emitter}',
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

    for hem in hemispheres:
        out_path = os.path.join(paths['output'], f'{batch}',
                                f'pocam_{device_id}',
                                f'hem_{hem}',
                                f'{emitter}',
                                'isotropy.json')
        #check is directory exists, if not create it
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, 'w+') as f:
            json.dump(isotropy, f, indent=4)
        print(f'  Saved → {out_path}')

    return isotropy