"""The result collector rejects incomplete or late DPF query material."""
import copy
import unittest
from run import validate_dpf_inventory


class DpfInventoryTests(unittest.TestCase):
    def inventory(self):
        plan=[dict(query=0,layer=8,slot=0,call=0,family=0,count=26,index_bits=8,
                   tables=2,output_bits=[16,16],grouped_tables=1),
              dict(query=0,layer=8,slot=0,call=1,family=1,count=13,index_bits=6,
                   tables=2,output_bits=[10,15],grouped_tables=0)]
        consumed=[dict(row,online_full_domain_eval=False,online_loaded_bytes=0,
                       online_generated=0) for row in plan]
        prepared=[dict(query=0,slot=0,requests=39,pre_input_eval_all=True,no_online_reload=True)]
        balance=[dict(query=0,generated=39,consumed=39,remaining=0)]
        phases=[dict(query=0,phase=phase,dpf_generation_epoch_delta=0)
                for phase in ("input","ready","online","output")]
        return [plan,prepared,consumed,balance,phases]

    def test_complete_material_ready_inventory_is_accepted(self):
        self.assertTrue(validate_dpf_inventory(*self.inventory(),1,True))

    def test_missing_material_and_reuse_are_rejected(self):
        for field in (0,1,2,3):
            rows=self.inventory();rows[field].pop()
            with self.assertRaises(ValueError):validate_dpf_inventory(*rows,1,True)
        rows=self.inventory();rows[2].append(copy.deepcopy(rows[2][0]))
        with self.assertRaises(ValueError):validate_dpf_inventory(*rows,1,True)

    def test_wrong_worker_and_shape_are_rejected(self):
        for field,value in (("slot",1),("index_bits",7),("count",25),("output_bits",[15,15])):
            rows=self.inventory();rows[2][0][field]=value
            with self.assertRaises(ValueError):validate_dpf_inventory(*rows,1,True)

    def test_online_generation_reload_and_eval_all_are_rejected(self):
        for field,value in (("online_generated",1),("online_loaded_bytes",1),("online_full_domain_eval",True)):
            rows=self.inventory();rows[2][0][field]=value
            with self.assertRaises(ValueError):validate_dpf_inventory(*rows,1,True)
        rows=self.inventory();rows[4][2]["dpf_generation_epoch_delta"]=1
        with self.assertRaises(ValueError):validate_dpf_inventory(*rows,1,True)

    def test_control_backend_cannot_hide_dpf_work(self):
        self.assertFalse(validate_dpf_inventory([],[],[],[],[],1,False))
        with self.assertRaises(ValueError):validate_dpf_inventory(*self.inventory(),1,False)

    def test_prepared_slot_duplicates_and_unready_material_are_rejected(self):
        rows=self.inventory();rows[1].append(dict(rows[1][0],requests=0))
        with self.assertRaises(ValueError):validate_dpf_inventory(*rows,1,True)
        rows=self.inventory();rows[1][0]["pre_input_eval_all"]=False
        with self.assertRaises(ValueError):validate_dpf_inventory(*rows,1,True)


if __name__=="__main__":unittest.main()
