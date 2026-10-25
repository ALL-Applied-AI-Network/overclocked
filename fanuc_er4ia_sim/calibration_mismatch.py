"""Deliberate assumed extrinsic errors; never modifies the rendered camera."""
import copy
import json
import math
import os
from pathlib import Path

ENV_NAMES = ('CAMERA_TRANSLATION_MM', 'CAMERA_ROTATION_DEG')


def configuration(env=None):
    env = os.environ if env is None else env
    values = []
    for name, allowed in zip(ENV_NAMES, ((0, 1, 2, 3), (0, 0.5))):
        try:
            value = float(env.get(name, '0'))
        except (ValueError, TypeError):
            raise ValueError(f'{name} must be one of {allowed}') from None
        if not math.isfinite(value) or value not in allowed:
            raise ValueError(f'{name} must be one of {allowed}')
        values.append(value)
    if all(values):
        raise ValueError('Choose translation OR rotation; vary one factor at a time')
    return dict(translation_mm=values[0], rotation_deg=values[1],
                translation_axis='+X', rotation_axis='+Z', frame='world',
                rotation_pivot='camera optical centre',
                relationship='assumed = true plus specified extrinsic error')


def environment(control):
    result = dict(zip(ENV_NAMES, (str(control['translation_mm']), str(control['rotation_deg']))))
    if configuration(result) != control:
        raise ValueError('Invalid recorded camera mismatch control')
    return result


def assumed_calibration(true, control):
    environment(control)  # Validate recorded semantics as well as magnitudes.
    assumed = copy.deepcopy(true)
    matrix = assumed['camera_to_world']
    matrix[3][0] += control['translation_mm'] / 1000.0
    angle = math.radians(control['rotation_deg'])
    c, s = math.cos(angle), math.sin(angle)
    # USD row vectors: rotate each world-space camera basis about world +Z.
    for row in matrix[:3]:
        x, y = row[:2]
        row[0], row[1] = c*x - s*y, s*x + c*y
    return assumed


def write_calibrations(run, true, control=None):
    run = Path(run)
    control = configuration() if control is None else control
    assumed = assumed_calibration(true, control)
    for name, data in (('camera_true.json', true), ('camera.json', assumed),
                       ('camera_mismatch.json', control)):
        with (run / name).open('x') as stream:
            json.dump(data, stream, indent=2)
            stream.write('\n')
