import collections,itertools,json,runpy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos'];inc=m['inc'];draws=m['draws'];combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}
committee_score=m.get('committee_score')
if committee_score is None:
 def committee_score(t):
  a1=m['a1_score'](t);a2=m['a2_score'](t);return a1,a2,m['z'](a1)+0.15*m['z'](a2)
def z(x):
 x=np.asarray(x,float);s=x.std();return (x-x.mean())/(s+1e-9)
def top_indices(score,k):
 idx=np.argpartition(-score,k-1)[:k];return idx[np.lexsort((idx,-score[idx]))]
def greedy(indices,K=10,triple_cap=1,pair_cap=2,num_cap=4,fallback=True):
 sel=[];tc=collections.Counter();pc=collections.Counter();nc=collections.Counter()
 for idx in indices:
  idx=int(idx);q=tuple(map(int,combos[idx]));trs=list(itertools.combinations(q,3));prs=list(itertools.combinations(q,2))
  if any(tc[x]>=triple_cap for x in trs) or any(pc[x]>=pair_cap for x in prs) or any(nc[x]>=num_cap for x in q):continue
  sel.append(idx)
  for x in trs:tc[x]+=1
  for x in prs:pc[x]+=1
  for x in q:nc[x]+=1
  if len(sel)>=K:break
 if fallback and len(sel)<K:
  used=set(sel)
  for idx in indices:
   idx=int(idx)
   if idx not in used:sel.append(idx);used.add(idx)
   if len(sel)>=K:break
 return sel
def final10(score,top):
 base=[]
 for c in (2,3,4):
  tr=greedy(top[:1500],10,1,2,c,False)
  if len(tr)>=10:base=tr[:10];break
 if len(base)<10:base=greedy(top[:1500],10,1,2,4,True)
 return base
def hist_pair(t,w=300):
 M=np.zeros((32,32),float)
 for row in draws[max(0,t-w):t]:
  for a,b in itertools.combinations(map(int,row),2):M[a,b]+=1;M[b,a]+=1
 return M
def select_vnext(t,stat,comm,base):
 tickets=[tuple(map(int,combos[i])) for i in base];union=sorted(set().union(*map(set,tickets)));bset=set(base);H=hist_pair(t,300)
 bp=set();bt=set()
 for q in tickets:bp.update(itertools.combinations(q,2));bt.update(itertools.combinations(q,3))
 rows=[]
 for q in itertools.combinations(union,5):
  idx=combo_index[q]
  if idx in bset:continue
  ovs=[len(set(q)&set(b)) for b in tickets];mo=max(ovs);parents=sum(x>0 for x in ovs);ps=list(itertools.combinations(q,2));ts=list(itertools.combinations(q,3));oldp=sum(p in bp for p in ps);oldt=sum(x in bt for x in ts);newp=10-oldp;newt=10-oldt;hp=sum(H[a,b] for a,b in ps)
  rows.append((idx,mo,parents,oldp,oldt,newp,newt,hp))
 idxs=np.array([r[0] for r in rows],int);F={k:z([r[j] for r in rows]) for k,j in [('mo',1),('parents',2),('oldp',3),('oldt',4),('newp',5),('newt',6),('hist',7)]};F['stat']=z(stat[idxs]);F['comm']=z(comm[idxs])
 scores={'core':.30*F['oldt']+.15*F['oldp']+.20*F['comm']+.20*F['stat']+.15*F['hist'],'pair':.20*F['oldp']+.20*F['parents']+.20*F['newp']+.15*F['comm']+.15*F['stat']+.10*F['hist'],'novel':.25*F['newp']+.25*F['newt']+.20*F['parents']+.15*F['comm']+.15*F['stat']}
 meta={r[0]:r for r in rows};chosen=[];used=set();covered=set();pc=collections.Counter();nc=collections.Counter();lanes=[('core',4,lambda r:r[1]>=3),('pair',3,lambda r:r[1]>=2),('novel',3,lambda r:True)]
 for name,K,pred in lanes:
  order=np.argsort(-scores[name],kind='stable');got=0
  while got<K:
   cand=[]
   for oi in order[:400]:
    idx=int(idxs[oi]);r=meta[idx];q=tuple(map(int,combos[idx]))
    if idx in used or not pred(r):continue
    if any(pc[p]>=2 for p in itertools.combinations(q,2)) or any(nc[n]>=4 for n in q):continue
    sc=float(scores[name][oi])+.35*sum(n not in covered for n in q);cand.append((sc,idx,q))
   if not cand:break
   _,idx,q=max(cand,key=lambda x:(x[0],-x[1]));chosen.append(idx);used.add(idx);covered.update(q)
   for p in itertools.combinations(q,2):pc[p]+=1
   for n in q:nc[n]+=1
   got+=1
 if len(chosen)<10:
  order=np.argsort(-F['comm'],kind='stable')
  for oi in order:
   idx=int(idxs[oi])
   if idx in used:continue
   chosen.append(idx);used.add(idx)
   if len(chosen)>=10:break
 return chosen[:10]
def metrics(ids,win):
 ws=set(win);ts=[set(map(int,combos[i])) for i in ids];u=set().union(*ts);P=set();T=set()
 for q in ts:P.update(itertools.combinations(sorted(q),2));T.update(itertools.combinations(sorted(q),3))
 return {'best':max(len(ws&q) for q in ts),'num':len(ws&u),'pair':len(set(itertools.combinations(sorted(ws),2))&P),'triple':len(set(itertools.combinations(sorted(ws),3))&T)}
def run(a,b):
 rows=[]
 for rr in range(a,b+1):
  t=rr-1;stat,_,comm=committee_score(t);top=top_indices(comm,5000);base=final10(comm,top);new=select_vnext(t,stat,comm,base);win=tuple(map(int,draws[t]));rows.append({'draw':rr,'winner':list(win),'base':metrics(base,win),'new':metrics(new,win)});print('done',rr,flush=True) if rr%25==0 else None
 return rows
def summ(rows):
 out={'n':len(rows)}
 for s in ('base','new'):out[s]={'3plus':sum(r[s]['best']>=3 for r in rows),'4plus':sum(r[s]['best']>=4 for r in rows),'5':sum(r[s]['best']==5 for r in rows),'num5':sum(r[s]['num']==5 for r in rows),'mean_num':float(np.mean([r[s]['num'] for r in rows])),'mean_pair':float(np.mean([r[s]['pair'] for r in rows])),'mean_triple':float(np.mean([r[s]['triple'] for r in rows]))}
 out['paired']={'gain3':sum(r['new']['best']>=3 and r['base']['best']<3 for r in rows),'loss3':sum(r['new']['best']<3 and r['base']['best']>=3 for r in rows),'gain4':sum(r['new']['best']>=4 and r['base']['best']<4 for r in rows),'loss4':sum(r['new']['best']<4 and r['base']['best']>=4 for r in rows),'gain5':sum(r['new']['best']==5 and r['base']['best']<5 for r in rows),'loss5':sum(r['new']['best']<5 and r['base']['best']==5 for r in rows)};return out
dev=run(1000,1199);test=run(1200,1399);pros=run(1400,min(1402,len(draws))) if len(draws)>=1400 else []
out={'definition':'Fixed Assembly-vNext: Core Preserve 4 + Pair Cross 3 + Novel 3; no sum floor or overlap hard cap; union coverage reward 0.35. No winner is used in generation. 1000-1199 development/reporting, 1200-1399 fixed test, 1400-1402 diagnostic only.','development':{'range':[1000,1199],'summary':summ(dev)},'test':{'range':[1200,1399],'summary':summ(test),'rows':test},'prospective_diagnostic':{'summary':summ(pros) if pros else None,'rows':pros},'warning':'This fixed rule was motivated after observing draw 1402, so the historical test is a robustness check, not independent confirmation of the hypothesis.'}
p=ROOT/'data'/'miniloto-assembly-vnext-fixed-audit.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'dev':summ(dev),'test':summ(test),'pros':summ(pros) if pros else None},ensure_ascii=False),flush=True)
