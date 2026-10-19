#!/usr/bin/env python3
"""Summarize one party's public trace, keeping nested primitive families separate."""
import argparse
import json
from pathlib import Path
from trace_sci import summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={}
    for group in summarize(args.input):
        total=result.setdefault(group['primitive'],dict(invocations=0,scalar_instances=0,widths={}))
        total['invocations']+=group['invocations']
        total['scalar_instances']+=group['scalar_instances']
        bits=str(group['input_bits'])
        total['widths'][bits]=total['widths'].get(bits,0)+group['scalar_instances']
    args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
