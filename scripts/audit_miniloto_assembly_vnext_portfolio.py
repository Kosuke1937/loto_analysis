import collections,itertools,json,runpy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos']; inc=m['inc']; draws=m['draws']; combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}
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
 support=inc[top[:500]].sum(0).astype(float);union=set().union(*[set(map(int,combos[i])) for i in base]);allowed=np.zeros(32,bool);allowed[list(union)]=True
 pool=top[allowed[combos[top]].all(axis=1)];pool=np.array([int(i) for i in pool if int(i) not in set(base)],int)
 if len(pool):
  zz=z(support[combos[pool]].mean(1));add=int(pool[np.argmax(score[pool]-0.10*zz)]);best=None;bk=None
  for rem in base:
   keep=[x for x in base if x!=rem]+[add];nu=set();pu=set()
   for idx in keep:
    q=tuple(map(int,combos[idx]));nu.update(q);pu.update(itertools.combinations(q,2))
   key=(len(nu),len(pu),float(np.sum(score[keep])))
   if bk is None or key>bk:bk=key;best=keep
  base=best
 return base
def hist_pair(t,w=300):
 M=np.zeros((32,32),float)
 for row in draws[max(0,t-w):t]:
  for a,b in itertools.combinations(map(int,row),2):M[a,b]+=1;M[b,a]+=1
 return M
def build_candidates(t,stat,comm,base,sum_cap=None):
 tickets=[tuple(map(int,combos[i])) for i in base];union=sorted(set().union(*map(set,tickets)));bset=set(base);H=hist_pair(t,300)
 bp=set();bt=set()
 for q in tickets:bp.update(itertools.combinations(q,2));bt.update(itertools.combinations(q,3))
 rows=[]
 for q in itertools.combinations(union,5):
  idx=combo_index[q]
  if idx in bset or (sum_cap is not None and sum(q)>sum_cap):continue
  overlaps=[len(set(q)&set(b)) for b in tickets];mo=max(overlaps);parents=sum(o>0 for o in overlaps)
  ps=list(itertools.combinations(q,2));ts=list(itertools.combinations(q,3));oldp=sum(p in bp for p in ps);oldt=sum(x in bt for x in ts);newp=10-oldp;newt=10-oldt
  hp=sum(H[a,b] for a,b in ps)
  rows.append((idx,mo,parents,oldp,oldt,newp,newt,hp))
 if not rows:return [],{},union
 idxs=np.array([r[0] for r in rows],int)
 vals={k:z([r[j] for r in rows]) for k,j in [('mo',1),('parents',2),('oldp',3),('oldt',4),('newp',5),('newt',6),('hist',7)]}
 vals['stat']=z(stat[idxs]);vals['comm']=z(comm[idxs])
 vals['core']=0.30*vals['oldt']+0.15*vals['oldp']+0.20*vals['comm']+0.20*vals['stat']+0.15*vals['hist']
 vals['pair']=0.20*vals['oldp']+0.20*vals['parents']+0.20*vals['newp']+0.15*vals['comm']+0.15*vals['stat']+0.10*vals['hist']
 vals['novel']=0.25*vals['newp']+0.25*vals['newt']+0.20*vals['parents']+0.15*vals['comm']+0.15*vals['stat']
 return rows,vals,union
def select_lane(rows,vals,base,counts=(4,3,3),union_reward=.35):
 idxs=np.array([r[0] for r in rows],int);meta={r[0]:r for r in rows};chosen=[];used=set();covered=set();pc=collections.Counter();nc=collections.Counter()
 lanes=[('core',counts[0],lambda r:r[1]>=3),('pair',counts[1],lambda r:r[1]>=2),('novel',counts[2],lambda r:True)]
 for lname,K,pred in lanes:
  score=vals[lname].copy();order=np.argsort(-score,kind='stable');got=0
  while got<K:
   cand=[]
   for oj in order[:min(400,len(order))]:
    j=int(idxs[oj]);rr=meta[j];qq=tuple(map(int,combos[j]))
    if j in used or not pred(rr):continue
    if any(pc[p]>=2 for p in itertools.combinations(qq,2)) or any(nc[n]>=4 for n in qq):continue
    a=float(score[oj])+union_reward*sum(n not in covered for n in qq);cand.append((a,j,qq))
   if not cand:break
   _,idx,q=max(cand,key=lambda x:(x[0],-x[1]));chosen.append(idx);used.add(idx);covered.update(q)
   for p in itertools.combinations(q,2):pc[p]+=1
   for n in q:nc[n]+=1
   got+=1
 if len(chosen)<10:
  order=np.argsort(-vals['comm'],kind='stable')
  for oi in order:
   idx=int(idxs[oi]);q=tuple(map(int,combos[idx]))
   if idx in used:continue
   chosen.append(idx);used.add(idx)
   if len(chosen)>=10:break
 return chosen[:10]
def metrics(ids,win):
 ws=set(win);ts=[set(map(int,combos[i])) for i in ids];u=set().union(*ts);pairs=set();triples=set()
 for q in ts:pairs.update(itertools.combinations(sorted(q),2));triples.update(itertools.combinations(sorted(q),3))
 wp=set(itertools.combinations(sorted(ws),2));wt=set(itertools.combinations(sorted(ws),3))
 return {'best':max(len(ws&q) for q in ts),'num':len(ws&u),'pair':len(wp&pairs),'triple':len(wt&triples)}
CONFIGS=[]
for cap in (None,105):
 for counts in ((4,3,3),(5,3,2),(3,3,4)):
  for ur in (.15,.35,.55):CONFIGS.append((cap,counts,ur))
def eval_draw(rr,cfg):
 t=rr-1;stat,_,comm=committee_score(t);top=top_indices(comm,5000);base=final10(comm,top);rows,vals,union=build_candidates(t,stat,comm,base,cfg[0]);sel=select_lane(rows,vals,base,cfg[1],cfg[2]);win=tuple(map(int,draws[t]));return {'draw':rr,'winner':list(win),'base':metrics(base,win),'new':metrics(sel,win),'base_union_size':len(set().union(*[set(map(int,combos[i])) for i in base])),'new_union_size':len(set().union(*[set(map(int,combos[i])) for i in sel]))}
def summary(rows):
 out={'n':len(rows)}
 for side in ('base','new'):
  out[side]={'3plus':sum(r[side]['best']>=3 for r in rows),'4plus':sum(r[side]['best']>=4 for r in rows),'5':sum(r[side]['best']>=5 for r in rows),'num5':sum(r[side]['num']==5 for r in rows),'mean_num':float(np.mean([r[side]['num'] for r in rows])),'mean_pair':float(np.mean([r[side]['pair'] for r in rows])),'mean_triple':float(np.mean([r[side]['triple'] for r in rows]))}
 out['paired']={'gain3':sum(r['new']['best']>=3 and r['base']['best']<3 for r in rows),'loss3':sum(r['new']['best']<3 and r['base']['best']>=3 for r in rows),'gain4':sum(r['new']['best']>=4 and r['base']['best']<4 for r in rows),'loss4':sum(r['new']['best']<4 and r['base']['best']>=4 for r in rows),'gain5':sum(r['new']['best']>=5 and r['base']['best']<5 for r in rows),'loss5':sum(r['new']['best']<5 and r['base']['best']>=5 for r in rows)}
 return out
scores=[]
for ci,cfg in enumerate(CONFIGS):
 rows=[eval_draw(rr,cfg) for rr in range(1000,1200)];s=summary(rows);key=(s['new']['5'],s['new']['4plus'],s['new']['3plus'],s['new']['num5'],s['new']['mean_pair'],s['new']['mean_triple'],-s['paired']['loss3']);scores.append((key,cfg,s));print('DEV',ci,cfg,s,flush=True)
best=max(scores,key=lambda x:x[0]);cfg=best[1]
print('SELECTED',cfg,best[2],flush=True)
test=[eval_draw(rr,cfg) for rr in range(1200,1400)];ts=summary(test)
pros=[]
for rr in (1400,1401,1402):
 if rr<=len(draws):pros.append(eval_draw(rr,cfg))
out={'definition':'Assembly-vNext portfolio audit. Fixed structural lane scores. Development 1000-1199 selects only sum cap / lane allocation / union reward. Fixed test 1200-1399 untouched. Draws 1400-1402 are diagnostic only and never used for selection.','lanes':{'core':'preserve existing parent triples/pairs + Stat/Committee/history','pair':'preserve pair + cross-parent/new-pair balance','novel':'new pair/triple exploration'},'selected_config':{'sum_cap':cfg[0],'lane_counts':cfg[1],'union_reward':cfg[2]},'development':{'range':[1000,1199],'selected_summary':best[2],'all_configs':[{'config':{'sum_cap':c[0],'lane_counts':c[1],'union_reward':c[2]},'summary':s} for _,c,s in scores]},'test':{'range':[1200,1399],'summary':ts,'rows':test},'prospective_diagnostic':pros,'warning':'No draw 1400-1402 result was used to select weights or configuration. This is historical model comparison, not evidence of future predictability.'}
p=ROOT/'data'/'miniloto-assembly-vnext-portfolio-audit.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print('TEST',ts,flush=True);print('WROTE',p)
