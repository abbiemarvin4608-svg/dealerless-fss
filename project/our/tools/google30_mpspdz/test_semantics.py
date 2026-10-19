"""Public integer adapter seam: worked sign/truncation examples and SCI fixtures."""
import unittest
from sci_semantics import signed, trunc0, matmul

class IntegerContractTests(unittest.TestCase):
    def test_signed_wrap_and_toward_zero_are_distinct_from_floor(self):
        self.assertEqual([signed(v, 8) for v in [127,128,255,256]], [127,-128,-1,0])
        self.assertEqual([trunc0(v, 3) for v in [-9,-8,-7,0,7,8,9]], [-1,-1,0,0,0,1,1])
    def test_sci_matmul_floors_wide_accumulator_before_rescaling(self):
        # K=2: sum=-1, internal floor(-1/2)=-1, then exact <<1.
        # This is deliberately not an ordinary dot product or SeeDot tree sum.
        self.assertEqual(matmul([1,1], [[-1],[0]], 8,8,16,0), [-2])
if __name__ == '__main__': unittest.main()
