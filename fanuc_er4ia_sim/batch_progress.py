"""Read-only live progress for an existing pipeline; never launches simulations."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import tempfile
import sys
import shutil
import threading
import time

STAGES = {
    'CONFIGURATION': 'Recording effective configuration',
    'CAPTURE': 'Isaac capture / simulator startup',
    'PERCEPTION': 'Cosmos startup / model loading',
    'PLANNING': 'Host planning',
    'EXECUTION': 'Physical execution / simulator startup / rendering',
    'VERIFY_REPORT': 'Checking execution report',
    'VIDEO_TABLE': 'Encoding table video',
    'VIDEO_SIDE': 'Encoding side video',
    'COMPLETE': 'Pipeline commands complete; awaiting batch outcome',
}


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def process_alive(pid):
    try:
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
            return False
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def clean(line):
    line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line)
    return ''.join(c for c in line if c.isprintable() or c == '\t')


class LogState:
    def __init__(self):
        self.offset = 0
        self.pending = b''
        self.stage = 'Waiting for pipeline log'
        self.stage_since = time.monotonic()
        self.last_output = self.stage_since
        self.pickups = set()
        self.summary = None

    def read(self, path):
        try:
            with path.open('rb') as stream:
                if path.stat().st_size < self.offset:
                    self.offset, self.pending = 0, b''
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return []
        if not chunk:
            return []
        self.last_output = time.monotonic()
        parts = (self.pending + chunk).replace(b'\r', b'\n').split(b'\n')
        self.pending = parts.pop()
        lines = [clean(line.decode('utf-8', errors='replace')) for line in parts]
        for line in lines:
            stage = None
            if line.startswith('[stage] '):
                stage = STAGES.get(line[8:].strip(), line[8:].strip())
            elif '[perceive] model ready' in line:
                stage = 'Perception: querying Cosmos on images'
            if stage is not None and stage != self.stage:
                self.stage, self.stage_since = stage, time.monotonic()
            match = re.search(r'\[sort\] detection (\d+):.*pickup=True', line)
            if match:
                self.pickups.add(match.group(1))
            match = re.search(r'\[sort\] (\d+)/6 pickups; (\d+) verified placements', line)
            if match:
                self.summary = f'{match[1]}/6 pickups; {match[2]}/6 placements reported by simulator'
        return [line for line in lines if line]


class PanelOutput:
    """Buffer each refresh and redraw it without accumulating terminal output."""
    def __init__(self):
        self.stream = sys.stdout
        self.interactive = self.stream.isatty() and os.environ.get('TERM') != 'dumb'
        self.rows = 0
        self.lines = []

    def write(self, value):
        self.lines.append(value)

    def flush(self):
        pass

    def render(self, final=False):
        width, height = shutil.get_terminal_size((80, 24))
        lines = ''.join(self.lines).splitlines()[:max(1, height - 2)]
        lines = [clean(line)[:max(1, width - 1)] for line in lines]
        self.lines.clear()
        if self.interactive:
            if self.rows:
                self.stream.write(f'\x1b[{self.rows}A')
            size = max(self.rows, len(lines))
            for index in range(size):
                self.stream.write('\r\x1b[2K' + (lines[index] if index < len(lines) else '') + '\n')
            self.rows = size
        elif final:
            self.stream.write('\n'.join(lines) + '\n')
        self.stream.flush()


def watch(log, run, index, count, seed, started, alive, interval=5):
    state = LogState()
    print(f'[progress] Run folder: {run}', flush=True)
    print(f'[progress] Full log: {log}', flush=True)
    print('[progress] Heartbeats show process existence, not guaranteed forward progress. No ETA is assumed.', flush=True)
    panel = PanelOutput()
    def panel_print(*args, **kwargs):
        print(*args, **kwargs, file=panel)
    while True:
        running = alive()
        lines = state.read(log)
        completed = index - 1
        filled = int(20 * completed / count)
        panel_print(f"Run {index}/{count} | seed {seed} | elapsed {duration(time.time()-started)} | PID {'exists' if running else 'ended'}", flush=True)
        panel_print(f"  Batch [{'#'*filled}{'.'*(20-filled)}] {completed}/{count} previous attempts finished (not success count)", flush=True)
        panel_print(f'  Stage: {state.stage} | observed stage time {duration(time.monotonic()-state.stage_since)}', flush=True)
        panel_print(f'  Last log activity: {duration(time.monotonic()-state.last_output)} ago', flush=True)
        panel_print('  ' + (state.summary or f'Pickups: {len(state.pickups)}/6; placements awaiting simulator summary'), flush=True)
        report_path = run / 'execution_report.json'
        try:
            report = json.loads(report_path.read_text())
            summary = (report.get('task_complete'), report.get('verified_pickups'), report.get('verified_placements'), report.get('abort_reason'))
            panel_print(f'  SAVED REPORT: task_complete={summary[0]}, pickups={summary[1]}, placements={summary[2]}, abort={summary[3]!r}', flush=True)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if not running:
            if state.pending:
                panel_print('  [last partial log line] ' + clean(state.pending.decode(errors='replace'))[:240], flush=True)
            panel_print('[progress] Pipeline process ended. Batch launcher will save its authoritative outcome.', flush=True)
            panel.render(final=True)
            return
        if not lines:
            panel_print('  Process exists; no new log output this refresh. This alone does not establish a stall or progress.', flush=True)
        panel.render()
        time.sleep(interval)


def demo():
    print('DEMO ONLY: synthetic log messages; no model, GPU or simulation is launched.', flush=True)
    with tempfile.TemporaryDirectory(prefix='overclocked_progress_demo_') as temp:
        run = Path(temp)
        log = run / 'demo.log'
        def producer():
            with log.open('w') as stream:
                for line in ('[stage] CAPTURE', '[capture] synthetic image saved', '[stage] PERCEPTION',
                             '[perceive] model ready', '[stage] PLANNING', '[stage] EXECUTION',
                             '[sort] detection 0: lifted synthetic object; pickup=True',
                             '[sort] 6/6 pickups; 6 verified placements; DEMO ONLY',
                             '[stage] VIDEO_TABLE', '[stage] VIDEO_SIDE', '[stage] COMPLETE'):
                    stream.write(line + '\n')
                    stream.flush()
                    time.sleep(0.3)
        worker = threading.Thread(target=producer)
        worker.start()
        watch(log, run, 1, 4, 'DEMO', time.time(), worker.is_alive, interval=0.3)
        worker.join()
    print('DEMO COMPLETE. All counts above were simulated display examples.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--index', type=int)
    parser.add_argument('--count', type=int)
    parser.add_argument('--seed')
    parser.add_argument('--started', type=float)
    parser.add_argument('--interval', type=float, default=5)
    args = parser.parse_args()
    if args.demo:
        demo()
        return
    if any(getattr(args, field) is None for field in ('log', 'run', 'pid', 'index', 'count', 'seed', 'started')):
        parser.error('Monitoring requires log, run, pid, index, count, seed and started.')
    if not 1 <= args.index <= args.count or not 0.2 <= args.interval <= 60 or args.pid < 1:
        parser.error('Invalid index/count/PID or interval; interval must be 0.2 to 60 seconds.')
    watch(args.log, args.run, args.index, args.count, args.seed, args.started,
          lambda: process_alive(args.pid), args.interval)


if __name__ == '__main__':
    main()
