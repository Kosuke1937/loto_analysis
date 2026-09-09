#!/usr/bin/env python3
from __future__ import annotations
import json,re
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)

def load():
 r=[]
 for p in sorted((ROOT/'data').glob('loto6-chunk-*.js'),key=lambda x:int(re.search(r'(\d+)',x.stem).group(1))):
  m=re.search(r'push\((.*)\);\s*$',p.read_text(encoding='utf-8'),re.S)
  if m:r.extend(json.loads(m.group(1)))
 r.sort(key=lambda x:x[0]);return [{'draw':x[0],'nums':list(map(int,x[2:8]))} for x in r]
def center(d):return sum(d['nums'])/6
def outside(H,t,n):
 p1=H[t-1]['nums'];p2=H[t-2]['nums'];p35=sum((H[t-k]['nums'] for k in (3,4,5)),[])
 return not(n in p1 or any(abs(n-x)==1 for x in p1) or any(abs(n-x)<=1 for x in p2) or n in p35)
def summary(rs):
 if not rs:return{}
 return {'n':len(rs),'mean_dist':sum(r['dist'] for r in rs)/len(rs),'mean_pull3':sum(r['pull3'] for r in rs)/len(rs),'pull3_positive_rate':sum(r['pull3']>0 for r in rs)/len(rs),'mean_pull5':sum(r['pull5'] for r in rs)/len(rs),'pull5_positive_rate':sum(r['pull5']>0 for r in rs)/len(rs),'closer3_rate':sum(r['closer3'] for r in rs)/len(rs),'closer5_rate':sum(r['closer5'] for r in rs)/len(rs)}
def main():
 H=load();start=len(H)-1000;end=len(H)-5;rec=[]
 for t in range(start,end):
  pre=sum(center(H[u]) for u in range(t-3,t))/3
  fut3=sum(center(H[u]) for u in range(t+1,t+4))/3
  fut5=sum(center(H[u]) for u in range(t+1,t+6))/5
  W=set(H[t]['nums'])
  for n in range(1,44):
   o=outside(H,t,n);y=n in W
   if not(y or o):continue
   typ='outside_winner' if o and y else 'outside_nonwinner' if o else 'inside_winner'
   d=n-pre;sg=1 if d>0 else -1 if d<0 else 0
   rec.append({'type':typ,'n':n,'dist':abs(d),'pull3':(fut3-pre)*sg,'pull5':(fut5-pre)*sg,'closer3':abs(fut3-n)<abs(pre-n),'closer5':abs(fut5-n)<abs(pre-n)})
 groups={g:[r for r in rec if r['type']==g] for g in ('outside_winner','outside_nonwinner','inside_winner')}
 ow=groups['outside_winner'];dists=sorted(r['dist'] for r in ow);q1=dists[len(dists)//4];q3=dists[3*len(dists)//4]
 out={'method':'Loto6 River Macro Pull v1','range':f"{H[start]['draw']}-{H[end-1]['draw']}",'groups':{g:summary(rs) for g,rs in groups.items()},'outside_distance_slices':{'near_q1':summary([r for r in ow if r['dist']<=q1]),'middle':summary([r for r in ow if q1<r['dist']<q3]),'far_q4':summary([r for r in ow if r['dist']>=q3])},'distance_thresholds':{'q1':q1,'q3':q3}}
 (OUT/'loto6_river_macro_pull_v1_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
