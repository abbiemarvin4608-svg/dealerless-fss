#!/usr/bin/env python3
"""Rebuild portable summaries/CSV from immutable per-party benchmark JSON."""
import argparse
import csv
import json
from pathlib import Path
from metrics import summarize_parties, distribution


def assemble(root):
    results=[];rows=[];profiles={}
    for path in sorted(root.glob('*/*/result.json')):
        result=json.loads(path.read_text());directory=path.parent
        if result['status']=='complete':
            parties=[json.loads((directory/f'party{role}_metrics.json').read_text()) for role in [1,2]]
            result.update(summarize_parties(parties))
        result['artifact']=str(path.relative_to(root));result['tag']=directory.name
        path.write_text(json.dumps(result,indent=2)+'\n');results.append(result)
        rows.append(dict(profile=result['network']['name'],run=directory.name,status=result['status'],queries=result['network'].get('queries',1),total_seconds=result['total_seconds'],first_query_seconds=result.get('first_query_seconds'),link_bytes=result['link_bytes'],online_seconds=result.get('online_seconds'),initialization_seconds=result.get('phases',{}).get('initialization_base_ot',{}).get('seconds'),session_seconds_per_query=result.get('amortization',{}).get('measured_session_seconds_per_query'),second_query_seconds=result.get('amortization',{}).get('measured_second_query_seconds'),peak_rss_party1_kib=(result.get('peak_rss_kib_per_party') or [None,None])[0],peak_rss_party2_kib=(result.get('peak_rss_kib_per_party') or [None,None])[1],dependent_rounds=None,reason=result.get('reason')))
    for name in sorted({r['network']['name'] for r in results}):
        selected=[r for r in results if r['network']['name']==name and r['tag'].startswith('repeat')]
        good=[r for r in selected if r['status']=='complete']
        profiles[name]=dict(completed=len(good),censored=sum(r['status']=='censored' for r in selected),latency_seconds=distribution([r['total_seconds'] for r in good]) if good else None,link_bytes=distribution([r['link_bytes'] for r in good]) if good else None)
    summary=dict(version='google30-sci-measurement-v2',profiles=profiles,runs=results)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if rows:
        with (root/'runs.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(assemble(a.output)['profiles'],indent=2))
