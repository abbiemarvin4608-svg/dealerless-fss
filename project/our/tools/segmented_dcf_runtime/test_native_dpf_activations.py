"""Exact frozen native SCI activation oracle versus public-table DPF hooks."""
import fcntl
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import tempfile
import unittest
from build import native_io_overlay, native_ot_overlays, native_overlays
from native_got_overlay import native_got_overlay
from native_planning_overlay import native_planning_overlays
from native_triple_overlay import native_triple_overlay
from run import records
HERE=Path(__file__).resolve().parent
SCI=Path('/home/jiang/EzPC/SCI')

class NativeDpfActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(prefix='native-dpf-activation-')
        out=Path(cls.directory.name)
        native_ot_overlays(out,SCI);native_io_overlay(out,SCI);native_got_overlay(out)
        native_planning_overlays(out,SCI);native_triple_overlay(out,SCI)
        names=('native_preprocessing.h','native_planning.h','native_plan_replay.h','native_query_materials.h',
            'prepared_got.h','prepared_cot.h','prepared_triples.h','measurement.h','io_counters.h',
            'backend_types.h','backend_adapter.h','round_trace.h','round_trace.cpp','native_workers.h',
            'native_workers.cpp','native_dpf_lut.h','native_dpf_lut.cpp','native_folded.h','native_folded.cpp','runtime_log.h','matrix_metrics.h',
            'test_native_dpf_activations.cpp')
        for name in names:(out/name).write_bytes((HERE/name).read_bytes())
        for name in ('sci_dpf_bridge.h','sci_msb_bridge.h','sci_folded_bridge.h'):(out/name).write_bytes((HERE.parent.parent/'src'/name).read_bytes())
        overrides={};native_overlays(out,SCI,overrides)
        cls.binary=out/'activation-test'
        bridge=Path(os.environ.get('SDCF_DPF_BRIDGE',str(HERE.parent.parent/'build-native-dpf/libsegfss_sci_bridge.so'))).resolve()
        command=['g++','-O3','-DNDEBUG','-std=c++17','-maes','-msse4.1','-mavx2','-mrdseed','-pthread','-fopenmp',
            '-DSCI_OT=1','-DEIGEN_DONT_PARALLELIZE','-Wno-deprecated-declarations',
            '-I'+str(out),'-I'+str(out/'include_override'),'-I'+str(SCI/'src'),'-I'+str(SCI/'extern/eigen')]
        command += [str(out/name) for name in ('test_native_dpf_activations.cpp','round_trace.cpp','native_workers.cpp','native_dpf_lut.cpp','native_folded.cpp')]
        command += [str(overrides.get(name,SCI/'src'/name)) for name in ('Math/math-functions.cpp','LinearOT/linear-ot.cpp',
            'BuildingBlocks/aux-protocols.cpp','BuildingBlocks/truncation.cpp','BuildingBlocks/value-extension.cpp')]
        command += ['-Wl,--no-as-needed',str(bridge),'-Wl,-rpath,'+str(bridge.parent),'-Wl,--start-group',
            str(SCI/'build/install/lib/libSCI-OT.a'),str(SCI/'build/install/lib/libSCI-GC.a'),'-Wl,--end-group',
            '-lssl','-lcrypto','-lgmp','-o',str(cls.binary)]
        with open('/tmp/sci-preprocessed-build.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);result=subprocess.run(command,text=True,capture_output=True)
        if result.returncode:raise AssertionError(result.stdout+result.stderr)

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def parties(self,mode,count,queries=2,exhaustive=False):
        for port in range(18000,57000,251):
            sockets=[]
            try:
                contexts=4 if os.environ.get('SDCF_GRAPH_MODE','sequential')!='sequential' else int(os.environ.get('SDCF_ACTIVATION_WORKERS','1'))
                offsets={0,50,100}
                for slot in range(contexts):
                    for base in (2000+150*slot, *((300+150*(slot-1),) if slot else ())):
                        offsets.update((base,base+50,base+100))
                for offset in sorted(offsets):
                    sock=socket.socket();sockets.append(sock);sock.bind(('127.0.0.1',port+offset))
                break
            except OSError:pass
            finally:
                for sock in sockets:sock.close()
        def limits(role):
            def apply():
                allowed=sorted(os.sched_getaffinity(0));cpus=(0,2,4,10) if role==1 else (6,8,12,16)
                selected=[cpu for cpu in cpus if cpu in allowed]
                os.sched_setaffinity(0,selected or allowed[(role-1)*4:role*4] or allowed[:4])
                resource.setrlimit(resource.RLIMIT_AS,(1536<<20,1536<<20))
            return apply
        outputs=[tempfile.TemporaryFile(mode='w+t') for _ in (1,2)]
        commands=[[str(self.binary),str(role),str(port),mode,str(count),str(queries)] for role in (1,2)]
        env=dict(os.environ,OMP_NUM_THREADS='2',SEGFSS_CGGM_THREADS='2',SDCF_ACTIVATION_EXHAUSTIVE='1' if exhaustive else '0')
        if not exhaustive:env.pop('SDCF_ACTIVATION_EXHAUSTIVE')
        children=[subprocess.Popen(commands[role-1],stdout=outputs[role-1],stderr=subprocess.STDOUT,text=True,
            preexec_fn=limits(role),env=env) for role in (1,2)]
        try:
            for child in children:child.wait(timeout=1200)
            logs=[]
            for output in outputs:output.seek(0);logs.append(output.read())
            if os.environ.get('SDCF_ACTIVATION_EVIDENCE_DIR'):
                evidence=Path(os.environ['SDCF_ACTIVATION_EVIDENCE_DIR'])/(mode+'_'+str(count)+'_'+str(queries)+('_full' if exhaustive else ''))
                evidence.mkdir(parents=True,exist_ok=True)
                for role,log in enumerate(logs,1):(evidence/('party%d.log'%role)).write_text(log)
                (evidence/'result.json').write_text(json.dumps({'mode':mode,'count':count,'queries':queries,'exhaustive':exhaustive,
                    'party_exit_codes':[c.returncode for c in children],'commands':commands,
                    'per_party_address_space_bytes':1536<<20,'max_cpu_per_party':4},indent=2)+'\n')
            self.assertEqual([c.returncode for c in children],[0,0],[log[-7000:] for log in logs])
            return logs
        finally:
            for child in children:
                if child.poll() is None:child.kill();child.wait()
            for output in outputs:output.close()

    def compare(self,count,queries=2,exhaustive=False,modes=('default','dpf_sigmoid','dpf_tanh','dpf')):
        reference=None
        for mode in modes:
            logs=self.parties(mode,count,queries,exhaustive)
            values=records(logs[1],'ACTIVATION_OUTPUT')
            self.assertEqual(len(values),2*queries)
            if reference is None:reference=values
            else:self.assertEqual(values,reference)
            if mode.startswith('dpf'):
                for log in logs:
                    rows=records(log,'DPF_LUT_BALANCE');self.assertEqual(len(rows),queries)
                    self.assertTrue(all(row['generated']==row['consumed'] and row['remaining']==0 for row in rows))
                    prep=records(log,'DPF_LUT_PREPARED');self.assertEqual(len(prep),queries)
                    self.assertTrue(all(row['pre_input_eval_all'] and row['no_online_reload'] for row in prep))
                    self.assertTrue(all(row.get('dpf_generation_epoch_delta')==0 for row in records(log,'METRIC')
                        if row['phase'] in ('input','ready','online','output')))

    def test_independent_hooks_match_native_boundary_integers(self):self.compare(25)
    def test_full_uint16_domain_matches_native_integer_oracle(self):self.compare(4096,16,True,('default','dpf'))
    def test_batch128_fresh_resident_activations_match_oracle(self):self.compare(12800,2,False,('default','dpf'))

if __name__=='__main__':unittest.main()
