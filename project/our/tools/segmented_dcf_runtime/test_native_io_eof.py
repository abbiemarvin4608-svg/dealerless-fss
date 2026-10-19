"""The real private SCI transport rejects a truncated incoming message."""
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
from build import native_io_overlay

SCI = Path('/home/jiang/EzPC/SCI')
HERE = Path(__file__).resolve().parent

class NativeIoEofTests(unittest.TestCase):
    def test_actual_netio_receive_rejects_eof_without_zero_progress_loop(self):
        with tempfile.TemporaryDirectory(prefix='native-netio-eof-') as directory:
            out = Path(directory)
            native_io_overlay(out, SCI)
            source = out / 'test.cpp'
            source.write_text('''#include "utils/net_io_channel.h"
int main(int argc, char** argv) {
  sci::NetIO io(nullptr, std::stoi(argv[1]), false, true);
  char value[4];
  try { io.recv_data(value, 4); }
  catch (const std::runtime_error&) { return 0; }
  return 1;
}
''')
            subprocess.run(['g++', '-O1', '-std=c++17', '-pthread', '-maes', '-msse4.1',
                '-I' + str(out / 'include_override'), '-I' + str(HERE),
                '-I' + str(SCI / 'src'), str(source), '-lssl', '-lcrypto',
                '-o', str(out / 'test')], check=True, capture_output=True)
            with socket.socket() as port_probe:
                port_probe.bind(('127.0.0.1', 0))
                port = port_probe.getsockname()[1]
            child = subprocess.Popen([str(out / 'test'), str(port)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 5
                while True:
                    try:
                        client = socket.create_connection(('127.0.0.1', port), timeout=1)
                        break
                    except ConnectionRefusedError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.01)
                with client:
                    client.sendall(b'abc')
                stdout, stderr = child.communicate(timeout=3)
                self.assertEqual(child.returncode, 0, stdout + stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()

if __name__ == '__main__':
    unittest.main()
