"""Apply recorded scene controls inside Isaac before importing target code."""
import json
import os
from pathlib import Path
import runpy
import sys
from yaw_config import ENV_NAMES, environment
import calibration_mismatch as calibration


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: isaac_run_entrypoint.py RUN_CONFIG_JSON SCRIPT [ARGS]')
    config = json.loads(Path(sys.argv[1]).read_text())['effective']
    script = Path(sys.argv[2]).resolve()
    for key in calibration.ENV_NAMES:
        os.environ.pop(key, None)
    os.environ.update(calibration.environment(config.get('camera_mismatch', calibration.configuration({}))))
    for key in ENV_NAMES:
        os.environ.pop(key, None)
    os.environ.update(environment(config['cube_yaw_control']))
    os.environ['CUBE_LAYOUT_SEED'] = str(int(config['cube_layout_seed']))
    import scene_config as C
    if C.SORT_CUBE_YAWS_DEG != config['cube_yaws_deg']:
        raise SystemExit('Recorded and simulator cube yaws disagree; target not loaded')
    actual_layout = [dict(name=f'SortCube{i}', x_m=x, y_m=y, colour=colour)
                     for i, (x, y, colour) in enumerate(C.SORT_CUBES)]
    if actual_layout != config['layout']:
        raise SystemExit('Recorded and simulator cube layout disagree; target not loaded')
    print('[configuration] Simulator layout and all six yaws match run_config.json', flush=True)
    sys.argv = [str(script), *sys.argv[3:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
