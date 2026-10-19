#!/usr/bin/env python3
"""Extract a native SCI recurrent step and compare its reconstructed tensors."""
import re
from pathlib import Path

SHAPES={'W1':512,'W2':1600,'U1':3500,'U2':3500,'Bg':100,'Bh':100}
SCALES={'tmp19':10,'tmp21':7,'tmp23':5,'tmp25':8,'tmp26':7,
        'tmp27_before_shl':2,'tmp27':7,'tmp28':14,'tmp29':7,'tmp30':7,
        'tmp31':16,'tmp33':14,'tmp34':17,'tmp35':17,'tmp36':17,'H':16}

def extract_calls(source):
    """Preserve every original argument, stopping before the classifier."""
    rows=[];counts={}
    pattern=r'^\s*(MatMul|MatAdd|MatSubBroadCastA|MatAddBroadCastB|Sigmoid|TanH|MulCir|ScalarMul|AdjustScaleShl)\(([^;]+)\);'
    for match in re.finditer(pattern,source,re.M):
        function=match[1];counts[function]=counts.get(function,0)+1
        if function=='MatMul' and counts[function]==5:break
        args=re.sub(r'\(\s*int(?:32|64)_t\s*\)','',match[2])
        args=[arg.strip() for arg in args.split(',')]
        output=args[-2] if function=='MatMul' else args[-1]
        bits=int(re.search(r'bw(\d+)$',output)[1])
        tensor=re.sub(r'bw\d+$','',output)
        if tensor=='tmp11':tensor='H'
        rows.append({'operator':f'{function}_{counts[function]:02}', 'function':function,
                     'tensor':tensor,'output':output,'bits':bits,'frac':SCALES[tensor],
                     'source_call':function+'('+', '.join(args)+');'})
    if len(rows)not in (15,16) or rows[-1]['tensor']!='H':
        raise ValueError('source does not contain the expected complete recurrent step')
    for i,row in enumerate(rows[:-1]):
        if rows[i+1]['function']=='AdjustScaleShl':
            row['tensor']='tmp27_before_shl';row['frac']=2
    # Independent artifact changes U2 scale, so the output is f10 before /8.
    if len(rows)==15:
        next(row for row in rows if row['tensor']=='tmp25')['frac']=10
    return rows

def compare_traces(sci,hybrid,schema,case_names):
    if set(sci)!=set(hybrid) or set(sci)!={str(i)for i in range(len(case_names))}:
        raise ValueError('missing or unexpected diagnostic case')
    first=None;count=0;total=0
    expected_tensors={row['tensor']for row in schema}
    for case_index,case in enumerate(case_names):
        a=sci[str(case_index)];b=hybrid[str(case_index)]
        if set(a)!=expected_tensors or set(b)!=expected_tensors:
            raise ValueError('missing tensor or unexpected tensor in '+case)
        for row in schema:
            tensor=row['tensor'];left=a[tensor];right=b[tensor]
            if len(left)!=len(right):raise ValueError('tensor length mismatch: '+tensor)
            for index,(x,y)in enumerate(zip(left,right)):
                total+=1
                if x!=y:
                    count+=1
                    if first is None:
                        first={'case':case,'case_index':case_index,'operator':row['operator'],
                               'tensor':tensor,'bits':row['bits'],'frac':row['frac'],
                               'element':index,'SCI':x,'SCI + ours':y}
    return {'equal':count==0,'differences':count,'output_values':total,'first_difference':first}

def read_values(path):
    cases={};
    for case,name,index,value in re.findall(r'^VALUE (\d+) (\w+) (\d+) (-?\d+)$',Path(path).read_text(),re.M):
        values=cases.setdefault(case,{}).setdefault(name,[])
        if int(index)!=len(values):raise ValueError('non-contiguous output '+name)
        values.append(int(value))
    return cases

def model_values(header,variant):
    text=Path(header).read_text();values={}
    narrow={'W1','U1'}|({'W2','U2','Bg'}if variant=='historical'else set())
    for name,count in SHAPES.items():
        match=re.search(r'\b'+name+r'_temp\s*\[[^=]+?=\s*\{(.*?)\};',text,re.S)
        if not match:raise ValueError('missing model tensor '+name)
        array=[int(x)for x in re.findall(r'-?\d+',match[1])]
        if len(array)!=count:raise ValueError('model shape mismatch '+name)
        if name in narrow:array=[x//256 if x>=0 else -((-x)//256)for x in array]
        values[name]=array
    return values
