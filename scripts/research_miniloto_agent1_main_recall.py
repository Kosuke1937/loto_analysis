import collections,itertools,json,runpy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos'];inc=m['inc'];draws=m['draws'];a1=m['a1_score'];a2=m['a2_score'];z=m['z']
DEV=range(1000,1200);TEST=range(1200,1400)

def committee(t): return z(a1(t))+0.15*z(a2(t))
def topidx(s,k=5000):
 i=np.argpartition(-s,k-1)[:k];return i[np.lexsort((i,-s[i]))]

def greedy(indices,K=10,triple_cap=1,pair_cap=2,num_cap=4,fallback=True):
 sel=[];tc=collections.Counter();pc=collections.Counter();nc=collections.Counter()
 for ii in indices:
  ii=int(ii);ns=tuple(map(int,combos[ii]));trs=list(itertools.combinations(ns,3));prs=list(itertools.combinations(ns,2))
  if any(tc[x]>=triple_cap for x in trs) or any(pc[x]>=pair_cap for x in prs) or any(nc[x]>=num_cap for x in ns): continue
  sel.append(ii)
  for x in trs:tc[x]+=1
  for x in prs:pc[x]+=1
  for x in ns:nc[x]+=1
  if len(sel)>=K:break
 if fallback and len(sel)<K:
  u=set(sel)
  for ii in indices:
   ii=int(ii)
   if ii not in u:sel.append(ii);u.add(ii)
   if len(sel)>=K:break
 return sel

def state_base(s,top):
 for cap in (2,3,4):
  q=greedy(top[:1500],10,1,2,cap,False)
  if len(q)>=10:return q[:10]
 return greedy(top[:1500],10,1,2,4,True)

def anti_support(s,top,base):
 support=inc[top[:500]].sum(0).astype(float);union=set()
 for ii in base:union.update(map(int,combos[ii]))
 allowed=np.zeros(32,dtype=bool);allowed[list(union)]=True
 pool=top[allowed[combos[top]].all(axis=1)];bset=set(base);pool=np.array([int(i) for i in pool if int(i) not in bset],dtype=int)
 if len(pool)==0:return base
 sup=support[combos[pool]].mean(1);zz=(sup-sup.mean())/(sup.std()+1e-9);add=int(pool[np.argmax(s[pool]-0.10*zz)])
 best=None;key0=None
 for rem in base:
  keep=[x for x in base if x!=rem]+[add];nu=set();pu=set()
  for ii in keep:
   ns=tuple(map(int,combos[ii]));nu.update(ns);pu.update(itertools.combinations(ns,2))
  key=(len(nu),len(pu),float(np.sum(s[keep])))
  if key0 is None or key>key0:key0=key;best=keep
 return best

def canonical(s,top):return anti_support(s,top,state_base(s,top))

def cov(s,top,lam):
 cand=list(map(int,top[:1500]));ss=s[cand];zs=(ss-ss.mean())/(ss.std()+1e-9);sel=[];u=set();pc=collections.Counter();tc=collections.Counter()
 for _ in range(10):
  bi=None;bv=-1e99
  for p,ii in enumerate(cand):
   if ii in sel:continue
   ns=tuple(map(int,combos[ii]));prs=list(itertools.combinations(ns,2));trs=list(itertools.combinations(ns,3))
   if any(pc[x]>=2 for x in prs) or any(tc[x]>=1 for x in trs):continue
   new=sum(n not in u for n in ns);v=float(zs[p])+lam*new
   if v>bv:bv=v;bi=ii
  if bi is None:break
  sel.append(bi);ns=tuple(map(int,combos[bi]));u.update(ns)
  for x in itertools.combinations(ns,2):pc[x]+=1
  for x in itertools.combinations(ns,3):tc[x]+=1
 if len(sel)<10:
  for ii in cand:
   if ii not in sel:sel.append(ii)
   if len(sel)>=10:break
 return sel[:10]

def metric(ids,t):
 u=set()
 for ii in ids:u.update(map(int,combos[ii]))
 w=set(map(int,draws[t]));return len(u&w),len(u),max(len(set(map(int,combos[ii]))&w) for ii in ids)

CACHE={}
for rr in list(DEV)+list(TEST):
 t=rr-1;s=committee(t);CACHE[rr]=(t,s,topidx(s))

def evaluate(period,method):
 rows=[]
 for rr in period:
  t,s,top=CACHE[rr]
  ids=canonical(s,top) if method=='canonical' else cov(s,top,float(method[3:])/100)
  h,us,b=metric(ids,t);rows.append((h,us,b))
 return {'n':len(rows),'union5':sum(h==5 for h,_,_ in rows),'union4plus':sum(h>=4 for h,_,_ in rows),'avg_union':round(float(np.mean([u for _,u,_ in rows])),3),'ticket3plus':sum(b>=3 for _,_,b in rows),'ticket4plus':sum(b>=4 for _,_,b in rows),'ticket5':sum(b==5 for _,_,b in rows)}

methods=['canonical','cov10','cov20','cov30','cov40','cov50','cov60','cov80','cov100']
dev={x:evaluate(DEV,x) for x in methods}
sel=max(dev,key=lambda x:(dev[x]['union5'],dev[x]['union4plus'],dev[x]['ticket4plus'],-dev[x]['avg_union']))
out={'protocol':{'dev':[1000,1199],'test':[1200,1399],'excluded':[1400,1401]},'development':dev,'selected':sel,'fixed_test':evaluate(TEST,sel),'canonical_test':evaluate(TEST,'canonical')}
(ROOT/'data'/'miniloto-agent1-main-recall.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
# trigger 2026-08-25T22:27+09:00
