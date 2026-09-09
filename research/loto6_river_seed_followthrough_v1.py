#!/usr/bin/env python3
from __future__ import annotations
import json,re,math
from pathlib import Path
from collections import defaultdict
from math import comb
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)

def load_history():
    rows=[]
    for p in sorted((ROOT/'data').glob('loto6-chunk-*.js'), key=lambda x:int(re.search(r'(\d+)',x.stem).group(1))):
        m=re.search(r'push\((.*)\);\s*$',p.read_text(encoding='utf-8'),re.S)
        if m: rows.extend(json.loads(m.group(1)))
    rows.sort(key=lambda r:r[0]); return [{'draw':r[0],'nums':list(map(int,r[2:8]))} for r in rows]

def outside(H,t,n):
    p1=H[t-1]['nums'];p2=H[t-2]['nums'];p35=sum((H[t-k]['nums'] for k in (3,4,5)),[])
    return not (n in p1 or any(abs(n-x)==1 for x in p1) or any(abs(n-x)<=1 for x in p2) or n in p35)

def silence(H,t,n,rad=2,cap=30):
    g=0
    for u in range(t-1,max(-1,t-cap-1),-1):
        if any(abs(x-n)<=rad for x in H[u]['nums']):return g
        g+=1
    return g

def width(n,rad): return sum(1 for x in range(1,44) if abs(x-n)<=rad)
def p_any_next(n,rad):
    m=width(n,rad); return 1-comb(43-m,6)/comb(43,6)
def future_stats(H,t,n,rad,w):
    cnt=0; anydraw=0; first=None
    for k in range(1,w+1):
        c=sum(abs(x-n)<=rad for x in H[t+k]['nums']); cnt+=c
        if c:
            anydraw+=1
            if first is None:first=k
    return cnt,anydraw,first

def summarize(rs,rad,w):
    if not rs:return {}
    cnt=sum(r[f'cnt{rad}_{w}'] for r in rs); any1=sum(r[f'any{rad}_1'] for r in rs if f'any{rad}_1' in r)
    expcnt=sum(w*6*width(r['n'],rad)/43 for r in rs)
    expany=sum(p_any_next(r['n'],rad) for r in rs) if w==1 else None
    return {'n':len(rs),'mean_count':cnt/len(rs),'random_expected_mean_count':expcnt/len(rs),'count_lift':cnt/expcnt if expcnt else None,
            **({'next_draw_any_rate':any1/len(rs),'random_expected_next_any':expany/len(rs),'next_any_lift':any1/expany if expany else None} if w==1 else {})}

def main():
    H=load_history(); start=len(H)-1000; end=len(H)-5
    rec=[]
    for t in range(start,end):
        W=set(H[t]['nums'])
        for n in range(1,44):
            o=outside(H,t,n); y=n in W
            if not (y or o): continue
            typ='outside_winner' if (o and y) else 'outside_nonwinner' if o else 'inside_winner'
            r={'draw':H[t]['draw'],'n':n,'type':typ,'silence2':silence(H,t,n,2)}
            for rad in (2,4):
                for w in (1,3,5):
                    cnt,anydraw,first=future_stats(H,t,n,rad,w);r[f'cnt{rad}_{w}']=cnt;r[f'anydraw{rad}_{w}']=anydraw
                    if w==1:r[f'any{rad}_1']=1 if cnt else 0
                    if w==5:r[f'first{rad}_5']=first
            rec.append(r)
    groups={g:[r for r in rec if r['type']==g] for g in ('outside_winner','inside_winner','outside_nonwinner')}
    out={'method':'Loto6 River Seed Follow-through v1','range':f"{H[start]['draw']}-{H[end-1]['draw']}",'groups':{},'outside_winner_silence2':{}}
    for g,rs in groups.items():
        out['groups'][g]={}
        for rad in (2,4):
            out['groups'][g][f'pm{rad}_next1']=summarize(rs,rad,1)
            out['groups'][g][f'pm{rad}_next3']=summarize(rs,rad,3)
            out['groups'][g][f'pm{rad}_next5']=summarize(rs,rad,5)
        first=[r['first2_5'] for r in rs if r['first2_5'] is not None]
        out['groups'][g]['pm2_first_within5']={'hit_within5_rate':len(first)/len(rs) if rs else 0,'mean_first_draw':sum(first)/len(first) if first else None}
    ow=groups['outside_winner']
    for key,sel in [('silence0-1',[r for r in ow if r['silence2']<=1]),('silence2',[r for r in ow if r['silence2']==2]),('silence3+',[r for r in ow if r['silence2']>=3])]:
        out['outside_winner_silence2'][key]={'n':len(sel),'pm2_next1':summarize(sel,2,1),'pm2_next5':summarize(sel,2,5)}
    path=OUT/'loto6_river_seed_followthrough_v1_summary.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
