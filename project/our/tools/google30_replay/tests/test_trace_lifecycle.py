"""The tracing CLI releases party processes when launch or execution fails."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'trace_sci.py'
SPEC = importlib.util.spec_from_file_location('trace_sci', SCRIPT)
TRACE_SCI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE_SCI)


class ControlledChild:
    """A process that waits for its peer until explicitly killed."""
    def __init__(self):
        self.returncode = None
        self.reaped = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise AssertionError('attempted to reap a child still waiting for its peer')
        self.reaped = True
        return self.returncode


class TraceLifecycleTest(unittest.TestCase):
    def run_cli(self, launch, sleep=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'runtime/SeeDot/temp/Predictor/input'
            inputs.mkdir(parents=True)
            for name in ['X.csv', 'Y.csv']:
                (inputs / name).write_text('0\n')
            argv = [str(SCRIPT), '--archive', str(root / 'archive'),
                    '--runtime', str(root / 'runtime'), '--sci', str(root / 'sci'),
                    '--output', str(root / 'output'), '--skip-build']
            with patch.object(sys, 'argv', argv), \
                    patch.object(TRACE_SCI.subprocess, 'Popen', side_effect=launch), \
                    patch.object(TRACE_SCI.time, 'sleep', side_effect=sleep):
                TRACE_SCI.main()

    def test_failed_second_launch_releases_first_party_and_both_logs(self):
        child = ControlledChild()
        logs = []

        def launch(*args, **kwargs):
            logs.append(kwargs['stdout'])
            if len(logs) == 2:
                raise OSError('second launch failed')
            return child

        with self.assertRaisesRegex(OSError, 'second launch failed'):
            self.run_cli(launch)

        self.assertIsNotNone(child.returncode, 'first party remained running')
        self.assertTrue(child.reaped, 'first party was not reaped')
        self.assertEqual(len(logs), 2)
        self.assertTrue(all(log.closed for log in logs), 'party logs remained open')

    def test_interrupted_launch_delay_releases_first_party_and_log(self):
        child = ControlledChild()
        logs = []

        def launch(*args, **kwargs):
            logs.append(kwargs['stdout'])
            return child

        with self.assertRaises(KeyboardInterrupt):
            self.run_cli(launch, KeyboardInterrupt)

        self.assertIsNotNone(child.returncode, 'first party remained running')
        self.assertTrue(child.reaped, 'first party was not reaped')
        self.assertEqual(len(logs), 1)
        self.assertTrue(logs[0].closed, 'first party log remained open')


if __name__ == '__main__':
    unittest.main()
