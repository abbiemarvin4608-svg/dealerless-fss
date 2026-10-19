"""Actual SCI sigmoid/tanh chains prepare all native material before query inputs."""
import fcntl
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import tempfile
import unittest

from build import native_io_overlay, native_ot_overlays
from native_got_overlay import native_got_overlay
from native_planning_overlay import native_planning_overlays
from native_triple_overlay import native_triple_overlay
from run import records, validate_native_materials

HERE = Path(__file__).resolve().parent
SCI = Path("/home/jiang/EzPC/SCI")


class PreparedActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="prepared-activation-")
        out = Path(cls.directory.name)
        native_ot_overlays(out,SCI); native_io_overlay(out,SCI)
        native_got_overlay(out); native_planning_overlays(out,SCI); native_triple_overlay(out,SCI)
        for name in ("native_preprocessing.h","native_planning.h","native_plan_replay.h","native_query_materials.h",
                     "prepared_got.h","prepared_cot.h","prepared_triples.h","measurement.h","io_counters.h",
                     "backend_types.h","backend_adapter.h","test_prepared_activations.cpp"):
            (out / name).write_bytes((HERE / name).read_bytes())
        cls.binary = out / "activation-test"
        command = ["g++","-O3","-DNDEBUG","-std=c++17","-maes","-msse4.1","-mavx2","-mrdseed",
                   "-pthread","-fopenmp","-DSCI_OT=1","-DEIGEN_DONT_PARALLELIZE","-Wno-deprecated-declarations",
                   "-I"+str(out),"-I"+str(out/"include_override"),"-I"+str(SCI/"src"),"-I"+str(SCI/"extern/eigen"),
                   str(out/"test_prepared_activations.cpp")]
        command += [str(SCI/"src"/name) for name in ("Math/math-functions.cpp","LinearOT/linear-ot.cpp",
                    "BuildingBlocks/aux-protocols.cpp","BuildingBlocks/truncation.cpp","BuildingBlocks/value-extension.cpp")]
        command += ["-Wl,--start-group",str(SCI/"build/install/lib/libSCI-OT.a"),
                    str(SCI/"build/install/lib/libSCI-GC.a"),"-Wl,--end-group",
                    "-lssl","-lcrypto","-lgmp","-o",str(cls.binary)]
        with open("/tmp/sci-preprocessed-build.lock","w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            result = subprocess.run(command,text=True,capture_output=True)
        if result.returncode: raise AssertionError(result.stdout+result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_parties(self,mode,count):
        for port in range(28731,46000,173):
            sockets = []
            try:
                for offset in (0,50,100):
                    sock=socket.socket(); sockets.append(sock); sock.bind(("127.0.0.1",port+offset))
                break
            except OSError: pass
            finally:
                for sock in sockets: sock.close()
        def limits(role):
            def apply():
                allowed=sorted(os.sched_getaffinity(0))
                selected=(0,2,4,10) if role==1 else (6,8,12,16)
                selected=[cpu for cpu in selected if cpu in allowed]
                os.sched_setaffinity(0,selected or allowed[(role-1)*4:role*4] or allowed[:4])
                resource.setrlimit(resource.RLIMIT_AS,(1536<<20,1536<<20))
            return apply
        outputs=[tempfile.TemporaryFile(mode="w+t") for _ in (1,2)]
        commands=[[str(self.binary),str(role),str(port),mode,str(count)] for role in (1,2)]
        children=[subprocess.Popen(commands[role-1],stdout=outputs[role-1],stderr=subprocess.STDOUT,text=True,
                  preexec_fn=limits(role),env=dict(os.environ,OMP_NUM_THREADS="2")) for role in (1,2)]
        try:
            for child in children: child.wait(timeout=120)
            logs=[]
            for output in outputs: output.seek(0); logs.append(output.read())
            if os.environ.get("SDCF_ACTIVATION_EVIDENCE_DIR"):
                evidence=Path(os.environ["SDCF_ACTIVATION_EVIDENCE_DIR"])/(mode+"_"+str(count))
                evidence.mkdir(parents=True,exist_ok=True)
                for role,log in enumerate(logs,1): (evidence/("party%d.log"%role)).write_text(log)
                (evidence/"result.json").write_text(json.dumps({"mode":mode,"count":count,
                    "party_exit_codes":[child.returncode for child in children],"commands":commands,
                    "per_party_address_space_bytes":1536<<20,"max_cpu_per_party":4},indent=2)+"\n")
            return [child.returncode for child in children],logs
        finally:
            for child in children:
                if child.poll() is None: child.kill(); child.wait()
            for output in outputs: output.close()

    def compare_activation_modes(self,count):
        reference=None
        for mode in ("default","prepared"):
            exits,logs=self.run_parties(mode,count)
            self.assertEqual(exits,[0,0],[log[-10000:] for log in logs])
            observed=records(logs[1],"ACTIVATION_OUTPUT")
            self.assertEqual(len(observed),4)
            self.assertTrue(all(len(row["values"])==count for row in observed))
            if reference is None: reference=observed
            else: self.assertEqual(observed,reference)
            if mode=="prepared":
                for log in logs:
                    valid,_=validate_native_materials(records(log,"MATERIAL_READY"),
                        records(log,"NATIVE_MATERIAL_PREPARED"),records(log,"NATIVE_MATERIAL_CONSUMED"),
                        records(log,"NATIVE_MATERIAL_BALANCE"),records(log,"METRIC"),2)
                    self.assertTrue(valid)
                    coverage=records(log,"ACTIVATION_MATERIAL_COVERAGE")
                    self.assertEqual(len(coverage),4)
                    self.assertTrue(all(row["native_ot_consumed"]>0 and row["bit_triples_consumed"]>0 and
                        row["native_generation_epoch_delta"]==row["triple_generation_epoch_delta"]==0 for row in coverage))
                    self.assertEqual(len(records(log,"ACTIVATION_PLAN")),2)
                    protocols={row["protocol"] for row in records(log,"NATIVE_PUBLIC_PLAN")}
                    self.assertTrue({"kkot_got","iknp_cot","iknp_batched_cot"}.issubset(protocols),protocols)
                    self.assertTrue(all(row["native_generation_epoch_delta"]==0 for row in records(log,"METRIC")
                        if row["phase"] in ("input","ready","online","output")))

    def test_native_activations_preserve_boundary_integers_and_irregular_tail(self):
        self.compare_activation_modes(13)

    def test_native_batch128_single_step_vectors_consume_all_fresh_material(self):
        self.compare_activation_modes(12800)

    def test_query_replay_rejects_missing_planned_triple_owner(self):
        exits,logs=self.run_parties("missing_owner",13)
        self.assertEqual(exits,[2,2],[log[-10000:] for log in logs])
        self.assertTrue(all("missing native triple replay context" in log for log in logs))
        self.assertTrue(all("MATERIAL_READY " not in log for log in logs))


if __name__=="__main__": unittest.main()
