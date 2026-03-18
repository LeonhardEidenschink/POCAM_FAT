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
from number_of_photons import compute_and_save


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
        batch = hem_entry['batch']

        for emitter in emitters:
            print(f'[hem {hemisphere}  |  device {device_id}  |  {emitter}]')
            try:
                result = compute_and_save(
                    hemisphere = hemisphere,
                    device_id  = device_id,
                    emitter    = emitter,
                    pwm        = s['pwm'],
                    temp       = s['temp'],
                    coarse     = s['coarse'],
                    fine       = s['fine'],
                    mode       = s['mode'],
                    target     = s['target'],
                    paths      = paths,
                )
                n     = result['meas_data'][0]['value']
                n_err = result['meas_data'][0]['error']
                print(f'  Photons: {n:.3e}  (rel. err: {n_err:.4f})')
                results[(hemisphere, emitter)] = result

            except Exception as e:
                print(f'  ERROR: {e}')

    print(f'\nDone. Processed {len(results)} / {total} combinations.')