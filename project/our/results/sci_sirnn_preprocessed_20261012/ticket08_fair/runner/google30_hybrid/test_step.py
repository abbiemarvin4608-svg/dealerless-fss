"""Public artifact seam: report the first integer mismatch, reject missing data."""
import unittest
from step import compare_traces, extract_calls

class StepContractTests(unittest.TestCase):
    def test_reports_first_operator_and_integer_mismatch(self):
        schema=[{'operator':'MatMul_1','tensor':'tmp19','bits':16,'frac':10}]
        left={'0':{'tmp19':[12,-1]}}
        right={'0':{'tmp19':[12,0]}}
        result=compare_traces(left,right,schema,['negative-truncation'])
        self.assertEqual(result['first_difference'],{'case':'negative-truncation','case_index':0,
          'operator':'MatMul_1','tensor':'tmp19','bits':16,'frac':10,'element':1,'SCI':-1,'SCI + ours':0})
        self.assertFalse(result['equal'])

    def test_missing_tensor_is_not_equal(self):
        schema=[{'operator':'Sigmoid_1','tensor':'tmp28','bits':16,'frac':14}]
        with self.assertRaisesRegex(ValueError,'missing tensor'):
            compare_traces({'0':{'tmp28':[8]}},{'0':{}},schema,['zero'])

if __name__=='__main__':unittest.main()
