"""Scheduled forks must only accept the exact approved FastGRNN graph."""
import unittest
from build import generate

# Original source calls, before public batch substitution/model preparation.
CALLS = [
    'MatMul(1,32,16,1,2,5,0,1,16,8,32,16,tmp15bw16,W1,tmp19bw16,tmp18bw32);',
    'MatMul(1,16,100,32,64,4,0,1,16,16,32,16,tmp19bw16,W2,tmp21bw16,tmp20bw32);',
    'MatMul(1,100,35,1,1,6,1,256,16,8,32,8,tmp11bw16,U1,tmp23bw8,tmp22bw32);',
    'MatMul(1,35,100,1,1,6,0,1,8,16,32,16,tmp23bw8,U2,tmp25bw16,tmp24bw32);',
    'MatAdd(1,100,1,8,1,1,16,16,16,16,tmp21bw16,tmp25bw16,tmp26bw16);',
    'MatAdd(1,100,1,8,1,1,16,16,16,16,tmp26bw16,Bg,tmp27bw16);',
    'Sigmoid(1,100,128,16384,16,16,tmp27bw16,tmp28bw16);',
    'MatAdd(1,100,1,8,1,1,16,16,16,16,tmp26bw16,Bh,tmp29bw16);',
    'TanH(1,100,128,128,16,16,tmp29bw16,tmp30bw16);',
    'MulCir(1,100,128,128,1,16,16,32,16,tmp28bw16,tmp11bw16,tmp31bw16);',
    'MatSubBroadCastA(1,100,1,1,1,1,16,16,16,16,tmp32,tmp28bw16,tmp33bw16);',
    'ScalarMul(1,100,128,128,1,16,16,32,16,tmp9bw16,tmp33bw16,tmp34bw16);',
    'MatAddBroadCastB(1,100,1,64,1,1,16,16,16,16,tmp34bw16,tmp10bw16,tmp35bw16);',
    'MulCir(1,100,8,16,1,16,16,32,16,tmp35bw16,tmp30bw16,tmp36bw16);',
    'MatAdd(1,100,1,2,1,1,16,16,16,16,tmp31bw16,tmp36bw16,tmp11bw16);',
]
CLASSIFIER = [
    'MatMul(1,100,30,1,1,1,1,1,16,8,32,8,tmp11bw16,FC,tmp39bw8,tmp38bw32);',
    'MatAdd(1,30,1,1,1,1,8,8,8,8,tmp39bw8,FCbias,tmp40bw8);',
]

class FrozenGraphTests(unittest.TestCase):
    def test_exact_frozen_calls_allow_whitespace(self):
        text, schema = generate('\n'.join(CALLS + CLASSIFIER))
        self.assertEqual(len(schema), 17)
        self.assertIn('sdcf_native::tail_parallel()', text)
        self.assertIn('sdcf_native::run_task_pair', text)

    def test_changed_tail_dependency_operator_shape_or_output_is_rejected(self):
        changes = [
            (9, 'tmp11bw16', 'tmp35bw16'),
            (10, 'MatSubBroadCastA', 'MatAddBroadCastB'),
            (11, '1,100,', '1,99,'),
            (12, 'tmp34bw16', 'tmp31bw16'),
            (13, 'tmp30bw16', 'tmp31bw16'),
            (14, 'tmp36bw16', 'tmp35bw16'),
            (9, 'tmp31bw16', 'tmp34bw16'),
            (1, ',64,4,', ',32,4,'),
        ]
        for index, before, after in changes:
            with self.subTest(index=index, after=after):
                calls = list(CALLS)
                calls[index] = calls[index].replace(before, after)
                with self.assertRaises(ValueError):
                    generate('\n'.join(calls + CLASSIFIER))

if __name__ == '__main__':
    unittest.main()
