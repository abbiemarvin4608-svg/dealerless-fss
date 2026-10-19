"""Private SCI activation overlays keep the original arithmetic around public LUT seams."""
from pathlib import Path
import tempfile
import unittest
from build import native_overlays

SCI = Path('/home/jiang/EzPC/SCI')

class NativeDpfOverlayTests(unittest.TestCase):
    def test_exact_public_functions_are_hooked_before_secret_message_tables(self):
        with tempfile.TemporaryDirectory(prefix='native-dpf-overlay-') as directory:
            out = Path(directory)
            overrides = {}
            native_overlays(out, SCI, overrides)
            self.assertIn('Math/math-functions.cpp', overrides)
            source = Path(overrides['Math/math-functions.cpp']).read_text()
            self.assertIn('sdcf_dpf::ActivationScope sdcf_activation(false)', source)
            self.assertIn('sdcf_dpf::ActivationScope sdcf_activation(true)', source)
            self.assertIn('sdcf_dpf::lookup_exp(this, dim, x_digits, digits_exp', source)
            self.assertIn('sdcf_dpf::lookup_reciprocal(this, dim, tmp_2, c0, c1, m, lookup_A0, lookup_A1)', source)
            self.assertIn('int idx = (tmp_2[i] + j) & m_mask;', source)
            self.assertIn('int idx = (x_digits[i] + j) & digit_mask;', source)
            self.assertIn('mult->hadamard_product(dim, c0, tmp_1, tmp_2', source)
            self.assertIn('trunc->truncate_and_reduce(dim, digits_exp + j * dim,', source)

if __name__ == '__main__': unittest.main()
