"""
calculate_self_monitoring.py
============================
Compute self-monitoring intensity profiles and photon numbers for all
hemispheres, emitters, and temperatures defined in a YAML config file.

Usage
-----
    python calculate_self_monitoring.py                          # uses self_monitoring_config.yaml
    python calculate_self_monitoring.py my_config.yaml          # or pass a custom config
"""

import sys
import yaml
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))  # Ensure lib/ is in the path for imports
from self_monitoring import compute_self_monitoring, save_emitter_json


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'self_monitoring_config.yaml'
    config      = load_config(config_path)

    s            = config['settings']
    emitters     = config['emitters']
    hemispheres  = config['hemispheres']
    temperatures = config['temperatures']
    paths        = config['paths']

    total   = len(hemispheres) * len(emitters) * len(temperatures)
    results = {}

    print(f'Config : {config_path}')
    print(f'Running {len(hemispheres)} hemisphere(s) × '
          f'{len(emitters)} emitter(s) × '
          f'{len(temperatures)} temperature(s) = {total} combination(s)\n')

    for hem_entry in hemispheres:
        hemisphere = hem_entry['hemisphere']
        device_id  = hem_entry['device_id']
        batch      = hem_entry['batch']

        for emitter in emitters:
            print(f'\n[hem {hemisphere} | device {device_id} | {emitter}]')
            meas_data_list = []
            meta = None

            for temp in temperatures:
                try:
                    entries, m = compute_self_monitoring(
                        hemisphere = hemisphere,
                        device_id  = device_id,
                        emitter    = emitter,
                        pwm        = s['pwm'],
                        temp       = temp,
                        coarse     = s['coarse'],
                        fine       = s['fine'],
                        mode       = s['mode'],
                        paths      = paths,
                        batch      = batch,
                        target     = s.get('target'),
                    )
                    meas_data_list.extend(entries)   # extend, not append
                    if meta is None:
                        meta = m
                    print(f'  temp={temp}°C → '
                        f'{entries[0]["value"]:.3e} photons '
                        f'(rel err: {entries[0]["rel_error"]:.4f})')
                except Exception as e:
                    print(f'  SKIP temp={temp}°C: {e}')

            if meas_data_list:
                save_emitter_json(hemisphere, device_id, emitter,
                                batch, meas_data_list, meta, paths)
            else:
                print('  WARNING: no successful measurements — no JSON written')