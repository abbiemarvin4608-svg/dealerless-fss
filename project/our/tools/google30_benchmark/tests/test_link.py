import importlib.util
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

SCRIPT=pathlib.Path(__file__).parents[1]/'link_proxy.py'

class LinkTest(unittest.TestCase):
    def test_real_tcp_bytes_survive_delay_and_rate_shaping(self):
        with socket.socket() as server:
            server.bind(('127.0.0.1',0)); server.listen(); target=server.getsockname()[1]
            with socket.socket() as reserve:
                reserve.bind(('127.0.0.1',0)); listen=reserve.getsockname()[1]
            payload=bytes(range(256))*257
            def echo():
                conn,_=server.accept()
                with conn:
                    while True:
                        data=conn.recv(65536)
                        if not data: break
                        conn.sendall(data)
            worker=threading.Thread(target=echo,daemon=True); worker.start()
            process=subprocess.Popen([sys.executable,str(SCRIPT),'--ports',f'{listen}:{target}','--rtt-ms','2','--mbps','100'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(),'READY')
                with socket.create_connection(('127.0.0.1',listen),timeout=5) as client:
                    started=time.monotonic(); client.sendall(payload); received=b''
                    while len(received)<len(payload): received+=client.recv(65536)
                    elapsed=time.monotonic()-started
                self.assertEqual(received,payload)
                self.assertGreaterEqual(elapsed,0.002)
                self.assertLess(elapsed,2)
            finally:
                process.terminate(); process.communicate(timeout=5); worker.join(timeout=5)

if __name__=='__main__': unittest.main()
