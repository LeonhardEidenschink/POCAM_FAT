"""
run_photons.py
==============
Compute the number of emitted photons per pulse for all hemispheres and
emitters defined in a YAML config file.

Usage
-----
    python run_photons.py                           # uses photons_config.yaml
    python run_photons.py my_config.yaml            # or pass a custom config
"""

import sys
import yaml
sys.path.append('lib')  # Ensure lib/ is in the path for imports
from number_of_photons import compute_photons, save_emitter_json


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'photons_config.yaml'
    config      = load_config(config_path)

    s           = config['settings']
    emitters    = config['emitters']
    hemispheres = config['hemispheres']
    paths       = config['paths']

    total   = len(hemispheres) * len(emitters)
    results = {}

    print(f'Config : {config_path}')
    print(f'Running {len(hemispheres)} hemisphere(s) × {len(emitters)} emitter(s) = {total} combination(s)\n')

    for hem_entry in hemispheres:
        hemisphere = hem_entry['hemisphere']
        device_id  = hem_entry['device_id']
        batch      = hem_entry['batch']

        for emitter in emitters:
            print(f'\n[hem {hemisphere} | device {device_id} | {emitter}]')
            meas_data_list = []
            meta = None

            # determine which combos apply to this emitter type
            if 'LMG' in emitter:
                combos = [
                    {'pwm': pwm, 'temp': temp, 'coarse': cf[0], 'fine': cf[1], 'mode': 'default', 'target': s['target']}
                    for pwm  in s['pwms']
                    for temp in s['temps']
                    for cf   in s['lmg_coarse_fine']
                ]
            else:
                combos = [
                    {'pwm': pwm, 'temp': temp, 'coarse': 1, 'fine': 20, 'mode': mode, 'target': s['target']}
                    for pwm  in s['pwms']
                    for temp in s['temps']
                    for mode in s['kapu_modes']
                ]

            for combo in combos:
                try:
                    entry, m = compute_photons(
                        hemisphere=hemisphere, device_id=device_id,
                        emitter=emitter, batch=batch, paths=paths, **combo)
                    meas_data_list.append(entry)
                    if meta is None:
                        meta = m   # use first successful measurement for top-level fields
                    print(f"  pwm={combo['pwm']} temp={combo['temp']}°C coarse={combo['coarse']} fine={combo['fine']} → "
                        f"{entry['value']:.3e} (err: {entry['error']:.4f})")
                except Exception as e:
                    print(f"  SKIP pwm={combo['pwm']} temp={combo['temp']}°C coarse={combo['coarse']} fine={combo['fine']}: {e}")

            if meas_data_list:
                save_emitter_json(hemisphere, device_id, emitter,
                                batch, meas_data_list, meta, paths)
            else:
                print('  WARNING: no successful measurements — no JSON written')