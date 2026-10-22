import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from batch_progress import LogState, watch


class ProgressTests(unittest.TestCase):
    def test_incremental_reads_keep_partial_lines_and_do_not_double_count_pickups(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'run.log'
            state = LogState()
            log.write_bytes(b'[stage] CAP')
            self.assertEqual(state.read(log), [])
            with log.open('ab') as stream:
                stream.write(b'TURE\n[sort] detection 3: lifted; pickup=True\n')
            state.read(log)
            self.assertIn('capture', state.stage)
            self.assertEqual(len(state.pickups), 1)
            self.assertEqual(state.read(log), [])
            with log.open('ab') as stream:
                stream.write(b'[sort] detection 3: lifted; pickup=True\n')
            state.read(log)
            self.assertEqual(len(state.pickups), 1)

    def test_final_drain_when_process_ends_and_no_false_success(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            log = run / 'run.log'
            log.write_text('[stage] PERCEPTION\nmodel ready\n')
            before = log.read_bytes()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                watch(log, run, 2, 4, '20261010', 0, lambda: False)
            self.assertIn('PID ended', output.getvalue())
            self.assertIn('1/4 previous attempts finished', output.getvalue())
            self.assertIn('awaiting simulator summary', output.getvalue())
            self.assertEqual(log.read_bytes(), before)
            self.assertEqual(list(run.iterdir()), [log])

    def test_missing_log_does_not_claim_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(LogState().read(Path(temp) / 'missing.log'), [])

    def test_carriage_return_progress_and_model_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'run.log'
            log.write_bytes(b'10%\r20%\r[perceive] model ready\n')
            state = LogState()
            lines = state.read(log)
            self.assertEqual(len(lines), 3)
            self.assertIn('querying Cosmos', state.stage)


if __name__ == '__main__':
    unittest.main()
