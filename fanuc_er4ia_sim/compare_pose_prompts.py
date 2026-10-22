"""Compare original/candidate Cosmos instructions on preserved image crops.

--prepare uses no GPU; --execute RUN_FOLDER performs inference only, never physics.
The production perception module is never edited by this tool.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import tempfile

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_RUNS = (
    'lieland_baseline_retry_dev_20261020_175604_DAoCi5',
    'lieland_default_updated_dev_20261021_205228_1E2CX4',
    'lieland_layout_20261010_rerun_dev_20261022_032116_alhxRN',
    'lieland_seed_20261010_dev_20261022_042332_13174_1_RGop4y',
    'lieland_seed_20261010_dev_20261022_042332_13174_2_TEKcxG',
    'lieland_seed_20261010_dev_20261022_042332_13174_3_d1VVB7',
    'lieland_seed_20261010_dev_20261022_042332_13174_4_bpoCg1',
)
CANDIDATE_PROMPT = '''Inspect the small cube in this camera crop and report its
colour and tabletop pose. A calibrated image-geometry front end has measured
its top-face orientation and centre from this SAME image, using the known cube
size and table camera calibration. These measurements are image-derived, not
simulator object coordinates. Use the supplied measurements for quantitative
pose; do not guess a new angle or regress pixel corners from this blurry crop.
Check that the object is a cube of the expected colour. If inconsistent or
ambiguous, report unknown and low confidence. Square yaw is modulo 90 degrees;
roll and pitch are constrained by the cube resting upright on the table.
Return only one JSON object with these fields:
- category: cube or unknown, based on the image.
- colour: yellow, blue, red, or unknown, based on the image.
- yaw_deg: a JSON number equal to the supplied measured yaw_deg, including its sign.
- pose_confidence: a JSON number from 0 to 1 reflecting your assessment.
- reason: a brief explanation of your visual assessment.
Use the measured yaw value from THIS query. Do not invent an angle or confidence.'''


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data, mode='x'):
    with Path(path).open(mode, encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_reply(answer, measurement):
    """Mirror the existing pose acceptance checks; never repair an answer."""
    if not isinstance(answer, dict):
        return {'accepted': False, 'reason': 'not_a_json_object'}
    try:
        reported = float(answer['yaw_deg'])
        confidence = float(answer.get('pose_confidence', 0))
    except (TypeError, ValueError, KeyError):
        return {'accepted': False, 'reason': 'missing_or_invalid_numeric_fields'}
    if not math.isfinite(reported) or not math.isfinite(confidence):
        return {'accepted': False, 'reason': 'nonfinite_reply'}
    disagreement = abs((reported - measurement['yaw_deg'] + 45) % 90 - 45)
    checks = {
        'category': answer.get('category') == 'cube',
        'colour': answer.get('colour') == measurement['expected_colour'],
        'confidence': 0.8 <= confidence <= 1,
        'yaw': disagreement <= 3,
    }
    return {'accepted': all(checks.values()), 'checks': checks,
            'square_yaw_disagreement_deg': disagreement,
            'reported_yaw_deg': reported, 'reported_confidence': confidence}


def prepare(root, source_names):
    from perceive import CUBE_POSE_PROMPT
    cases, excluded = [], []
    for name in source_names:
        if Path(name).name != name:
            raise ValueError('Source run must be a folder name, not a path.')
        source = root / 'runs' / name
        folder = source / 'cube_pose_crops'
        measurements = sorted(folder.glob('cube_*_measurement.json'))
        if not measurements:
            raise FileNotFoundError(f'No saved cube measurements: {folder}')
        for path in measurements:
            stem = path.name.removesuffix('_measurement.json')
            image, raw = folder / (stem + '.png'), folder / (stem + '.txt')
            measurement = json.loads(path.read_text())
            if float(measurement['silhouette_iou']) < 0.85:
                excluded.append({'run': name, 'cube': stem, 'reason': 'original_image_fit_below_0.85'})
                continue
            for required in (image, raw):
                if not required.is_file():
                    raise FileNotFoundError(required)
            if not math.isfinite(float(measurement['yaw_deg'])):
                raise ValueError(f'Nonfinite image measurement: {path}')
            cases.append({
                'case_id': f'case_{len(cases):03d}', 'source_run': name, 'cube': stem,
                'image': str(image.resolve()), 'image_sha256': digest(image),
                'measurement_file': str(path.resolve()), 'measurement_sha256': digest(path),
                'historical_reply_file': str(raw.resolve()), 'historical_reply_sha256': digest(raw),
                'measurement': measurement,
            })
    if not cases:
        raise ValueError('No eligible saved crops.')
    output = Path(tempfile.mkdtemp(prefix='lieland_pose_compare_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_', dir=root / 'runs'))
    (output / 'cube_pose_crops').mkdir()
    config = {
        'schema_version': 1, 'experiment_class': 'development',
        'condition': 'saved_crop_pose_prompt_comparison', 'run_id': output.name,
        'started_at_utc': utc(), 'source_sha256': {
            name: digest(root / name) for name in ('perceive.py', 'cube_pose.py', 'compare_pose_prompts.py')},
        'prompts': {'original': CUBE_POSE_PROMPT, 'candidate': CANDIDATE_PROMPT},
        'suffix': '\nMeasured image geometry for THIS cube:\n',
        'max_new_tokens': 400, 'gpu': 0,
        'case_count': len(cases), 'inference_calls': 2 * len(cases),
        'cases': cases, 'excluded': excluded,
        'comparison_order': 'Alternate original-first and candidate-first by case index.',
        'limits': 'One paired comparison per crop; correlated development examples, not a formal success rate. No physics or grasp commands. Ground truth not read.',
    }
    write_json(output / 'run_config.json', config)
    write_json(output / 'development_outcome.json', {
        'experiment_class': 'development', 'run_id': output.name,
        'outcome': 'prepared_no_inference', 'case_count': len(cases),
        'inference_calls': 2 * len(cases), 'task_complete': None,
        'verified_pickups': None, 'verified_placements': None,
    })
    print(f'PREPARED_FOLDER={output}')
    print(f'{len(cases)} saved crops; {2 * len(cases)} planned model calls. No model loaded or simulations launched.')


def execute(output, root):
    from perceive import ask, load_model, parse_json
    config = json.loads((output / 'run_config.json').read_text())
    if config['condition'] != 'saved_crop_pose_prompt_comparison':
        raise ValueError('Not a prepared prompt comparison folder.')
    for name, expected in config['source_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Source changed since preparation: {name}. Prepare a NEW comparison.')
    for case in config['cases']:
        for field, hash_field in (('image', 'image_sha256'), ('measurement_file', 'measurement_sha256'), ('historical_reply_file', 'historical_reply_sha256')):
            if digest(case[field]) != case[hash_field]:
                raise ValueError(f'Saved input changed: {case[field]}')
    # Exclusive creation refuses reruns/resume so old replies cannot be overwritten.
    with (output / 'inference_started.txt').open('x') as stream:
        stream.write(utc() + '\n')
    results, state, error = [], 'inference_running', None
    start = utc()
    def save():
        write_json(output / 'development_outcome.json', {
            'schema_version': 1, 'experiment_class': 'development', 'run_id': output.name,
            'outcome': state, 'started_at_utc': start, 'updated_at_utc': utc(),
            'case_count': config['case_count'], 'planned_calls': config['inference_calls'],
            'completed_calls': len(results), 'results': results, 'error': error,
            'task_complete': None, 'verified_pickups': None, 'verified_placements': None,
            'limits': config['limits'],
        }, 'w')
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Terminated by signal {signum}')
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        save()
        model, processor = load_model(config['gpu'])
        for index, case in enumerate(config['cases']):
            order = ('original', 'candidate') if index % 2 == 0 else ('candidate', 'original')
            for variant in order:
                prompt = config['prompts'][variant] + config['suffix'] + json.dumps(case['measurement'])
                stem = output / 'cube_pose_crops' / (case['case_id'] + '_' + variant)
                with stem.with_suffix('.prompt.txt').open('x') as stream:
                    stream.write(prompt)
                print(f"[comparison] {len(results)+1}/{config['inference_calls']} {case['source_run']} {case['cube']} {variant}", flush=True)
                raw = ask(model, processor, Path(case['image']), prompt, config['max_new_tokens'])
                with stem.with_suffix('.txt').open('x') as stream:
                    stream.write(raw)
                validation = validate_reply(parse_json(raw), case['measurement'])
                result = {'case_id': case['case_id'], 'source_run': case['source_run'],
                          'cube': case['cube'], 'variant': variant,
                          'validation': validation, 'raw_reply_file': str(stem.with_suffix('.txt').relative_to(output))}
                write_json(stem.parent / (stem.name + '_measurement.json'), {
                    'input_measurement': case['measurement'], 'result': result})
                results.append(result)
                save()
        state = 'saved_crop_comparison_complete'
    except BaseException as exc:
        state = 'comparison_interrupted' if isinstance(exc, KeyboardInterrupt) else 'comparison_error'
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
        signal.signal(signal.SIGTERM, previous)
        print(f'COMPARISON_FOLDER={output}', flush=True)
    for variant in ('original', 'candidate'):
        subset = [r for r in results if r['variant'] == variant]
        print(f"{variant}: {sum(r['validation']['accepted'] for r in subset)}/{len(subset)} replies passed existing pose checks")
    print('COMPARISON FINISHED. Saved-crop model evidence only; no physics tested.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', type=Path)
    parser.add_argument('--source-run', action='append', help='Folder name under runs; repeat to override the documented default sources.')
    args = parser.parse_args()
    if args.prepare:
        prepare(ROOT, args.source_run or DEFAULT_SOURCE_RUNS)
    else:
        if args.source_run:
            parser.error('--source-run is for preparation only')
        execute(args.execute.resolve(), ROOT)


if __name__ == '__main__':
    main()
