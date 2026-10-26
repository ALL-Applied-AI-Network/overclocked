"""Apply recorded scene controls inside Isaac before importing target code."""
import json
import os
from pathlib import Path
import runpy
import sys
import faulthandler
from yaw_config import ENV_NAMES, environment
import calibration_mismatch as calibration


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: isaac_run_entrypoint.py RUN_CONFIG_JSON SCRIPT [ARGS]')
    config_path = Path(sys.argv[1]).resolve()
    config = json.loads(config_path.read_text())['effective']
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
    # Startup probes passed with Python output flushing enabled and also
    # stalled without it. Candidate workaround; full-pipeline validation needed.
    # Configure both streams before Kit can create logging threads.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(line_buffering=True, write_through=True)
    sys.argv = [str(script), *sys.argv[3:]]
    sys.path.insert(0, str(script.parent))
    trace = None
    if script.name == 'sort_scene.py':
        trace = (Path(config_path).parent / 'execution_startup_traceback.txt').open('a')
        faulthandler.dump_traceback_later(120, repeat=True, file=trace)
        print('[execution startup] Traceback watchdog armed: 120 seconds; diagnostics only', flush=True)
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        if trace is not None:
            faulthandler.cancel_dump_traceback_later()
            trace.close()


if __name__ == '__main__':
    main()
