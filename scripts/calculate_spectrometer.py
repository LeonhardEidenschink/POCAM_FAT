"""
run_spectrometer.py
===================
Compute spectral profiles for all hemispheres, emitters, and temperatures
defined in a YAML config file.

Usage
-----
    python run_spectrometer.py                          # uses spectrometer_config.yaml
    python run_spectrometer.py my_config.yaml           # or pass a custom config
"""

import sys
import yaml
sys.path.append('./lib')  # Ensure the spectrometer module can be imported
from spectrometer import compute_and_save


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'spectrometer_config.yaml'
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
        batch = hem_entry['batch']

        for emitter in emitters:
            for temp in temperatures:
                print(f'[hem {hemisphere}  |  device {device_id}  '
                      f'|  {emitter}  |  {temp}°C]')
                try:
                    result = compute_and_save(
                        hemisphere = hemisphere,
                        device_id  = device_id,
                        emitter    = emitter,
                        pwm        = s['pwm'],
                        temp       = temp,
                        coarse     = s['coarse'],
                        fine       = s['fine'],
                        mode       = s['mode'],
                        paths      = paths,
                        batch = batch,
                    )
                    cwl      = result['meas_data'][2]['value']
                    cwl_err  = result['meas_data'][2]['error']
                    fwhm     = result['meas_data'][1]['value']
                    fwhm_err = result['meas_data'][1]['error']
                    print(f'  CWL:  {cwl:.2f} ± {cwl_err:.2f} nm')
                    print(f'  FWHM: {fwhm:.2f} ± {fwhm_err:.2f} nm')
                    results[(hemisphere, emitter, temp)] = result

                except Exception as e:
                    print(f'  ERROR: {e}')

    print(f'\nDone. Processed {len(results)} / {total} combinations.')