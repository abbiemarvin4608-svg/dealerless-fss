#!/usr/bin/env python3
"""Compare reconstructed diagnostic tensors from identical native and Beaver runs."""
import argparse
import json
from pathlib import Path


def tensors(path):
    result=[]
    for line in Path(path).read_text().splitlines():
        if not line.startswith("TENSOR "):continue
        parts=line.split();descriptor=parts[1:7];values=list(map(int,parts[7:]))
        if len(values)!=int(parts[5])*int(parts[6]):raise ValueError("truncated tensor record")
        result.append((descriptor,values))
    if not result:raise ValueError("no diagnostic tensors")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--native",type=Path,required=True);p.add_argument("--beaver",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    native,beaver=tensors(a.native),tensors(a.beaver)
    mismatches=[]
    if len(native)!=len(beaver):mismatches.append({"reason":"tensor_count"})
    for (desc,x),(other,y) in zip(native,beaver):
        if desc!=other:mismatches.append({"reason":"tensor_descriptor","native":desc,"beaver":other});continue
        for index,(left,right) in enumerate(zip(x,y)):
            if left!=right:
                mismatches.append({"tensor":desc,"index":index,"sample":index//int(desc[5]),"column":index%int(desc[5]),"SCI":left,"Beaver":right});break
    report={"status":"passed" if not mismatches else "failed","tensors":len(native),"values":sum(len(v) for _,v in native),"mismatches":mismatches,"scope":"one Batch=128 timestep, all graph-level reconstructed integer outputs"}
    a.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
    if mismatches:raise SystemExit(1)
