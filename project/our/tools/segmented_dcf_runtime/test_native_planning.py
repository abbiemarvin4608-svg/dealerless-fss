"""Public shape traversal must preserve real protocol state and private results."""
import fcntl
import os
from pathlib import Path
import resource
import socket
import subprocess
import tempfile
import unittest

from build import native_io_overlay, native_ot_overlays
from native_planning_overlay import native_planning_overlays

HERE = Path(__file__).resolve().parent
SCI = Path("/home/jiang/EzPC/SCI")


class NativePlanningTests(unittest.TestCase):
    def test_public_b2a_plan_generates_fresh_stock_without_reading_choices(self):
        with tempfile.TemporaryDirectory(prefix="native-planning-") as directory:
            out = Path(directory)
            native_ot_overlays(out, SCI)
            native_io_overlay(out, SCI)
            native_planning_overlays(out, SCI)
            binary = out / "planning-test"
            command = ["g++", "-O2", "-std=c++17", "-maes", "-msse4.1", "-mavx2", "-mrdseed", "-pthread", "-fopenmp", "-DSCI_OT=1",
                       "-Wno-deprecated-declarations", "-I" + str(out), "-I" + str(out / "include_override"), "-I" + str(HERE),
                       "-I" + str(SCI / "src"), str(HERE / "test_native_planning.cpp"),
                       str(SCI / "src/BuildingBlocks/aux-protocols.cpp"),
                       "-Wl,--start-group", str(SCI / "build/install/lib/libSCI-OT.a"),
                       str(SCI / "build/install/lib/libSCI-BuildingBlocks.a"),
                       str(SCI / "build/install/lib/libSCI-GC.a"), "-Wl,--end-group",
                       "-lssl", "-lcrypto", "-lgmp", "-o", str(binary)]
            with open("/tmp/sci-preprocessed-build.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for port in range(26500, 46000, 173):
                sockets = []
                try:
                    for offset in (0, 50, 100):
                        sock = socket.socket(); sockets.append(sock); sock.bind(("127.0.0.1", port + offset))
                    break
                except OSError:
                    pass
                finally:
                    for sock in sockets: sock.close()
            def limits(role):
                def apply():
                    cpus = sorted(os.sched_getaffinity(0))
                    os.sched_setaffinity(0, cpus[(role - 1) * 4:role * 4])
                    resource.setrlimit(resource.RLIMIT_AS, (1610612736, 1610612736))
                return apply
            children = [subprocess.Popen([str(binary), str(role), str(port)], stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, preexec_fn=limits(role)) for role in (1, 2)]
            try:
                logs = [child.communicate(timeout=45)[0] for child in children]
                self.assertEqual([child.returncode for child in children], [0, 0], logs)
                for log in logs:
                    self.assertEqual(log.count('"side_effects":0'), 2)
                    self.assertEqual(log.count('"consumed":11'), 2)
                    self.assertEqual(log.count('"requested":11,"generated":512'), 4)
            finally:
                for child in children:
                    if child.poll() is None: child.kill(); child.wait()


if __name__ == "__main__":
    unittest.main()
