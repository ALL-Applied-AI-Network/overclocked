"""Validated cube yaw controls in degrees; independent of layout randomness."""
import math
import os
import random

ORIGINAL_YAWS_DEG = (0.0, 26.0, 32.0, -23.0, 14.0, -36.0)
ENV_NAMES = ('CUBE_YAW_MODE', 'CUBE_YAW_DEG', 'CUBE_YAW_MAX_DEG', 'CUBE_YAW_SEED')


def angle(value, name):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'{name} must be a finite number from 0 to 45 degrees') from None
    if not math.isfinite(number) or not 0 <= number <= 45:
        raise ValueError(f'{name} must be a finite number from 0 to 45 degrees')
    return number


def resolve(environ=None):
    env = os.environ if environ is None else environ
    mode = env.get('CUBE_YAW_MODE', 'original')
    if mode not in ('original', 'fixed', 'random'):
        raise ValueError('CUBE_YAW_MODE must be original, fixed, or random')
    allowed = {'original': (), 'fixed': ('CUBE_YAW_DEG',),
               'random': ('CUBE_YAW_MAX_DEG', 'CUBE_YAW_SEED')}[mode]
    unexpected = [key for key in ENV_NAMES[1:] if key in env and key not in allowed]
    if unexpected:
        raise ValueError(f'{", ".join(unexpected)} is not used in {mode} yaw mode; unset it')
    settings = dict(mode=mode, fixed_deg=None, max_deg=None, seed=None)
    if mode == 'original':
        return settings, list(ORIGINAL_YAWS_DEG)
    if mode == 'fixed':
        settings['fixed_deg'] = angle(env.get('CUBE_YAW_DEG'), 'CUBE_YAW_DEG')
        return settings, [settings['fixed_deg']] * 6
    settings['max_deg'] = angle(env.get('CUBE_YAW_MAX_DEG', '45'), 'CUBE_YAW_MAX_DEG')
    try:
        settings['seed'] = int(env.get('CUBE_YAW_SEED', '20261009'))
    except (ValueError, TypeError):
        raise ValueError('CUBE_YAW_SEED must be an integer') from None
    rng = random.Random(settings['seed'])
    return settings, [rng.uniform(0, settings['max_deg']) for _ in range(6)]


def environment(settings):
    """Convert a recorded effective control to its canonical environment."""
    result = {'CUBE_YAW_MODE': settings['mode']}
    if settings['mode'] == 'fixed':
        result['CUBE_YAW_DEG'] = str(settings['fixed_deg'])
    elif settings['mode'] == 'random':
        result['CUBE_YAW_MAX_DEG'] = str(settings['max_deg'])
        result['CUBE_YAW_SEED'] = str(settings['seed'])
    # Revalidate canonical settings before use.
    resolve(result)
    return result
