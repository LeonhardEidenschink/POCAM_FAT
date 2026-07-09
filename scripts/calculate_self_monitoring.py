"""
run_self_monitoring.py
=======================
Reads self_monitoring_config.yaml and runs compute_and_save() for every
(hemisphere x emitter x temperature) combination listed in it.

Usage
-----
    python run_self_monitoring.py [path/to/config.yaml]

If no path is given, looks for 'self_monitoring_config.yaml' in the
current directory.
"""

import argparse
import glob
import os
import sys

import yaml
sys.path.append('./lib/')
from self_monitoring import compute_and_save


# ---------------------------------------------------------------------------
# Path-building helpers
#
# NOTE: These two functions encode assumptions about your file-naming
# conventions that I could not verify against your actual data directories.
# Please check them against a real example of each before trusting the
# output, and adjust the glob patterns below if they don't match.
# ---------------------------------------------------------------------------

def find_sm_file(data_sm_dir, temp):
    """
    Locate the self-monitoring HDF5 file for a given temperature inside
    data_sm_dir.

    ASSUMPTION: the file name contains '{temp}C' (e.g. a file named
    something like 'self_monitoring_25C.h5' or 'run_-20C.h5'). Adjust the
    pattern below if your files are named differently.
    """
    pattern = os.path.join(data_sm_dir, f'*_{temp}C*.h5')
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f'No self-monitoring HDF5 file found in {data_sm_dir!r} '
            f'matching pattern {pattern!r}.'
        )
    if len(matches) > 1:
        raise RuntimeError(
            f'Multiple self-monitoring HDF5 files matched {pattern!r} in '
            f'{data_sm_dir!r}: {matches}. Narrow the pattern in '
            f'find_sm_file() so exactly one file is selected.'
        )
    return matches[0]


def resolve_flasher_key(emitter, hemisphere, flasher_key_map):
    """
    Resolve a device model name (e.g. 'LMG405') to the hardware flasher-slot
    key used inside the self-monitoring HDF5 file (e.g. 'lmg1', 'kapu0').

    IMPORTANT: which physical LED sits in which slot is hemisphere/device
    specific. `flasher_key_map` is read from the config's optional
    `flasher_keys` section, keyed as '{hemisphere}:{emitter}' (checked
    first) or, as a fallback, just '{emitter}' if every hemisphere uses the
    same slot for that model. Neither of these is guessed — if the mapping
    isn't present in the config, this raises loudly so you don't silently
    process the wrong flasher slot.

    Your original code imported _lmg_key / _kapu_key / _resolve_target from
    pocam_utils, which likely already do this resolution correctly (probably
    by reading it out of the HDF5 metadata itself). If so, replace this
    function's body with a call to that instead of maintaining a config map.
    """
    key = f'{hemisphere}:{emitter}'
    if key in flasher_key_map:
        return flasher_key_map[key]
    if emitter in flasher_key_map:
        return flasher_key_map[emitter]
    raise KeyError(
        f'No flasher_key mapping found for emitter={emitter!r} '
        f'hemisphere={hemisphere!r}. Add an entry to the config\'s '
        f'"flasher_keys" section, e.g.:\n'
        f'  flasher_keys:\n'
        f'    "{hemisphere}:{emitter}": lmg1   # or kapu0, etc.'
    )


def find_abs_cal_glob(data_abs_cal_dir, hemisphere, emitter, device_id, batch, T):
    """
    Build the glob pattern for absolute-calibration JSON files for a given
    hemisphere.

    Per the config comment, cal_prefix is "joined with hemisphere id" —
    i.e. files are expected to look like '{cal_prefix}{hemisphere}...json'
    (there can be several, one per power/pwm setting).
    """
    if 'LMG' in emitter:
        fname = f'photons_*_{T}C_1-20.json'
    if 'KAPU' in emitter:
        fname = f'photons_*_{T}C_default.json'
    return os.path.join(
        data_abs_cal_dir, batch, f'pocam_{device_id}', f'hem_{hemisphere}', emitter, fname)


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = config['paths']
    settings = config.get('settings', {})
    temperatures = config['temperatures']
    emitters = config['emitters']
    hemispheres = config['hemispheres']

    data_abs_cal_dir = paths['data_abs_cal']
    output_dir = paths['output']
    cal_prefix = paths.get('cal_prefix', 'cali_flange_')
    flasher_key_map = config.get('flasher_keys', {})

    n_total = len(hemispheres) * len(emitters) * len(temperatures)
    n_done = 0
    n_failed = 0
    failures = []

    print(f'Starting run: {len(hemispheres)} hemispheres x {len(emitters)} '
          f'emitters x {len(temperatures)} temperatures = {n_total} jobs\n')
    
    emitters = [('lmg1','LMG365'),('lmg0','LMG405'), ('lmg3','LMG450'), ('lmg2','LMG520'), ('kapu0','KAPU405'), ('kapu3','KAPU465')]

    for hem_entry in hemispheres:
        hemisphere = str(hem_entry['hemisphere'])
        device_id = str(hem_entry['device_id'])
        batch = hem_entry['batch']

        data_sm_dir = paths['data_sm'].format(batch=batch, hem=hemisphere)

        for emitter_touple in emitters:
            emitter = emitter_touple[1]
            flasher =  emitter_touple[0]
            for temp in temperatures:
                job_desc = (f'hem={hemisphere} device={device_id} '
                            f'emitter={emitter} temp={temp}C')
                try:
                    sm_filepath = find_sm_file(data_sm_dir, temp)
                    abs_cal_glob = find_abs_cal_glob(
                        data_abs_cal_dir, hemisphere, emitter, device_id, batch, temp
                    )

                    compute_and_save(
                        hemisphere=hemisphere,
                        device_id=device_id,
                        emitter=emitter,
                        flasher_key=flasher,
                        temp=temp,
                        sm_filepath=sm_filepath,
                        abs_cal_glob=abs_cal_glob,
                        output_dir=output_dir,
                        batch=batch,
                        settings=settings,
                    )
                    n_done += 1
                except Exception as exc:
                    n_failed += 1
                    failures.append((job_desc, repr(exc)))
                    print(f'  FAILED [{job_desc}]: {exc}')

    print(f'\nDone: {n_done}/{n_total} succeeded, {n_failed} failed.')
    if failures:
        print('\nFailed jobs:')
        for job_desc, err in failures:
            print(f'  - {job_desc}: {err}')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'config', nargs='?', default='self_monitoring_config.yaml',
        help='Path to the YAML config file (default: ./self_monitoring_config.yaml)'
    )
    args = parser.parse_args()
    main(args.config)