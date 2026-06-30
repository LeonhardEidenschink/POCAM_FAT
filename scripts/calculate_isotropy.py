"""
run_isotropy.py
===============
Runs isotropy computation for all devices and emitters defined in a YAML config file.

Usage
-----
    python run_isotropy.py                          # uses isotropy_config.yaml by default
    python run_isotropy.py my_config.yaml           # or pass a custom config path
"""

import sys
import yaml
sys.path.append('lib')  # Ensure lib/ is in the path for imports
from isotropy import compute_and_save


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'isotropy_config.yaml'
    config      = load_config(config_path)

    s        = config['settings']
    paths    = config['paths']
    emitters = config['emitters']
    devices  = config['devices']

    total    = len(devices) * len(emitters)
    results  = {}

    print(f'Config : {config_path}')
    print(f'Running {len(devices)} device(s) × {len(emitters)} emitter(s) = {total} combination(s)\n')

    for device in devices:
        device_id   = device['id']
        hemispheres = device['hemispheres']
        batch       = device['batch']
        print('running for device:', device_id)

        for emitter in emitters:
            print(f'[{device_id}  |  {emitter}  |  {batch}]')
            try:
                result = compute_and_save(
                    device_id   = device_id,
                    hemispheres = hemispheres,
                    batch       = batch,
                    emitter     = emitter,
                    coarse      = s['coarse'],
                    fine        = s['fine'],
                    mode        = s['mode'],
                    pwm         = s['pwm'],
                    temp        = s['temp'],
                    target      = s['target'],
                    paths       = paths,
                )
                isotropy_value_central_68 = result['meas_data'][1]['value_central_68']
                isotropy_value_min_max= result['meas_data'][1]['value_min_max']
                iso_err = result['meas_data'][1]['error']
                print(f'  Isotropy: {isotropy_value_min_max:.3f} ± {iso_err:.3f} (min-max range)')
                print(f'  Isotropy: {isotropy_value_central_68:.3f} (center of central quantile)')
                results[(device_id, emitter)] = result

            except Exception as e:
                print(f'  ERROR: {e}')

    print(f'\nDone. Processed {len(results)} / {total} combinations.')