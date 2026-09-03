#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h',ROOT/'research'/'loto6_historical_rebuild_rank_audit.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
p=h.p;a=h.a
DRAWS=[2112,2113,2117,2118,2120,2121,2122,2125,2126,2127,2128,2129,2130,2131,2133]
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
def main():
 rows=p.fetch_history();rows=[r for r in rows if r[0]<=2133];di={d:i for i,(d,_,_) in enumerate(rows)};draws=np.asarray([r[1] for r in rows],np.int16);bonus=np.asarray([r[2] for r in rows],np.int16)
 Cref=p.fixed_sample();stref,inc,qcuts=p.build_static(Cref);_,_,npref,ppref,actual=p.hist_actual(rows,qcuts);sizes,priors=p.prepare_priors(stref)
 out=[]
 for draw in DRAWS:
  t=di[draw];win=tuple(map(int,draws[t]));recent20=set(map(int,draws[t-20:t].ravel()));pool21,sref,c5ref,c4ref,pairc=h.support_pool21(t,Cref,stref,inc,draws,bonus,npref,ppref,actual,sizes,priors,recent20)
  c20=Counter(a.band(r) for r in draws[t-20:t]);c2150=Counter(a.band(r) for r in draws[t-50:t-20]);c50=Counter(a.band(r) for r in draws[t-50:t]);prev=set(map(int,draws[t-1]));valid=[]
  for c in itertools.combinations(pool21,6):
   sh=a.band(c)
   if c20[sh]>0 or not (c2150[sh]==1 or c50[sh]==0) or len(set(c)&prev)>1:continue
   valid.append(c)
  inpool=all(x in pool21 for x in win);rec={'draw':draw,'winner':list(win),'sum':sum(win),'pool21':list(pool21),'winner_in_pool21':inpool,'valid_count':len(valid),'winner_prev_overlap':len(set(win)&prev)}
  if inpool and rec['winner_prev_overlap']<=1:
   W=p.weights(t,500,actual,sizes,priors);X,fx=a.direct_static(valid,qcuts);_,_,_,vcomm=a.direct_score(t,X,fx,W,draws,bonus,npref,pairc,sref,c5ref,c4ref);WX,wfx=a.direct_static([win],qcuts);_,_,_,wc=a.direct_score(t,WX,wfx,W,draws,bonus,npref,pairc,sref,c5ref,c4ref);rank=int(np.count_nonzero(vcomm>wc[0])+1);rec.update({'committee_score':float(wc[0]),'rank':rank,'rank_pct':rank/len(valid)*100,'equiv_rank_40882':int(round(rank/len(valid)*40882))})
  else:rec.update({'committee_score':None,'rank':None,'rank_pct':None,'equiv_rank_40882':None})
  out.append(rec);print(draw,rec,flush=True)
 Path(OUT/'loto6_historical_rebuild_rank_last15.json').write_text(json.dumps({'rows':out},ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
