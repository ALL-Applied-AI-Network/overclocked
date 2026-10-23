"""Saved-crop inventory comparison. Never edits production or starts physics."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path('/workspace/lieland/fanuc_er4ia_sim')
sys.path.insert(0, str(ROOT))
RUNS = (
    'lieland_seed_20261010_dev_20261023_182630_17307_1_IX81vR',
    'lieland_seed_20261010_dev_20261023_182630_17307_2_cR5X6i',
)
CANDIDATE = '''Classify the object nearest the CENTRE of this crop using its visible
geometry and colour. A cube is a small block with straight edges and square flat
faces, possibly rounded corners. A cap has a broad circular flange and a
cylindrical plug or hollow cup. Yellow cubes can look beige/tan.
Return ONLY one JSON object. Include ALL THREE required fields in every reply:
- category: cube, cap, or unknown, based on visible geometry.
- colour: yellow, blue, red, orange, another visible colour, or unknown.
- confidence: a JSON number from 0 to 1 expressing your assessment of the
  category and colour together. Assess it from this image; do not use a preset.
If the geometry or colour is ambiguous, report unknown and low confidence.
Do not infer a colour from the task or invent missing objects. Check that
category, colour, AND numeric confidence are all present before replying.
Do not add prose outside the JSON object.'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def validate(answer, colour):
    # Match the production inventory gate exactly; never fill missing fields.
    if not isinstance(answer, dict):
        return False
    try:
        return (answer.get('category') == 'cube' and answer.get('colour') == colour
                and float(answer.get('confidence', 0)) >= 0.8)
    except (ValueError, TypeError):
        return False


def prepare():
    tree = ast.parse((ROOT / 'perceive.py').read_text())
    prompts = [node.value.value for node in ast.walk(tree)
               if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
               and any(isinstance(t, ast.Name) and t.id == 'detail_prompt' for t in node.targets)]
    if len(prompts) != 1:
        raise ValueError('Could not uniquely identify current inventory prompt.')
    cases = []
    for run in RUNS:
        folder = ROOT / 'runs' / run
        for image in sorted((folder / 'cube_inventory_crops').glob('red_candidate*.png')):
            # Human visual review of saved crops; never passed to the model.
            should_accept = image.stem in ('red_candidate3', 'red_candidate4')
            if image.stem not in ('red_candidate1', 'red_candidate3', 'red_candidate4'):
                raise ValueError(f'Unreviewed crop: {image}')
            cases.append(dict(image=str(image), sha256=digest(image), expected_colour='red',
                              should_accept=should_accept, group='inventory', source_run=run))
        for i in range(4):
            image = folder / 'cube_pose_crops' / f'cube_{i}.png'
            cases.append(dict(image=str(image), sha256=digest(image),
                              expected_colour='yellow' if i < 2 else 'blue',
                              should_accept=True, group='colour_control', source_run=run))
    if len(cases) != 14:
        raise ValueError(f'Expected 14 saved crops, found {len(cases)}')
    output = Path(tempfile.mkdtemp(prefix='lieland_pose_compare_inventory_' +
                  datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_', dir=ROOT / 'runs'))
    (output / 'cube_pose_crops').mkdir()
    config = dict(experiment_class='development', condition='saved_crop_inventory_comparison',
                  cases=cases, prompts=dict(original=prompts[0], candidate=CANDIDATE),
                  source_sha256=digest(ROOT / 'perceive.py'), tool_sha256=digest(__file__),
                  max_new_tokens=300, planned_calls=28,
                  limits='Correlated development crops, not a formal success rate. No physics or ground-truth use.')
    save(output / 'run_config.json', config)
    save(output / 'development_outcome.json', dict(outcome='prepared_no_inference', planned_calls=28))
    print('PREPARED_FOLDER=' + str(output))
    print('14 crops; 28 planned Cosmos calls. No model loaded; production unchanged.')


def execute(output):
    from perceive import ask, canonicalise, load_model, parse_json
    config = json.loads((output / 'run_config.json').read_text())
    if config['condition'] != 'saved_crop_inventory_comparison':
        raise ValueError('Wrong comparison type')
    if config['source_sha256'] != digest(ROOT / 'perceive.py') or config['tool_sha256'] != digest(__file__):
        raise ValueError('Source changed; prepare a new comparison')
    for case in config['cases']:
        if digest(case['image']) != case['sha256']:
            raise ValueError('Saved crop changed')
    with (output / 'inference_started.txt').open('x') as stream:
        stream.write(datetime.now(timezone.utc).isoformat())
    results = []
    state = 'running'
    error = None
    try:
        model, processor = load_model(0)
        for index, case in enumerate(config['cases']):
            order = ('original', 'candidate') if index % 2 == 0 else ('candidate', 'original')
            for variant in order:
                stem = output / 'cube_pose_crops' / f'case_{index:03d}_{variant}'
                prompt = config['prompts'][variant]
                stem.with_suffix('.prompt.txt').write_text(prompt)
                print(f'[inventory comparison] {len(results)+1}/28 {Path(case["image"]).name} {variant}', flush=True)
                raw = ask(model, processor, Path(case['image']), prompt, 300)
                stem.with_suffix('.txt').write_text(raw)
                answer = canonicalise(parse_json(raw) or {})
                accepted = validate(answer, case['expected_colour'])
                result = dict(case_index=index, variant=variant, accepted=accepted,
                              correct_gate_decision=accepted == case['should_accept'],
                              answer=answer, group=case['group'])
                save(stem.parent / (stem.name + '_measurement.json'), dict(case=case, result=result))
                results.append(result)
                save(output / 'development_outcome.json', dict(outcome=state, results=results))
        state = 'saved_crop_inventory_comparison_complete'
    except BaseException as exc:
        state, error = 'comparison_error_or_interruption', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save(output / 'development_outcome.json', dict(experiment_class='development', outcome=state,
             completed_calls=len(results), planned_calls=28, results=results, error=error,
             task_complete=None, verified_pickups=None, verified_placements=None))
        print('COMPARISON_FOLDER=' + str(output), flush=True)
    for variant in ('original', 'candidate'):
        subset = [r for r in results if r['variant'] == variant]
        print(f'{variant}: {sum(r["correct_gate_decision"] for r in subset)}/14 correct gate decisions')
        print(f'  accepted {sum(r["accepted"] for r in subset)}; expected 12 accepts and 2 orange rejections')
    print('MODEL COMPARISON COMPLETE. Production unchanged; no physics tested.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        execute(args.execute.resolve())
