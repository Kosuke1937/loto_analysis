#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p',ROOT/'research'/'loto6_consensus_phase1_sample.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
PREV_BONUS_EXCLUDE=18

def local_rows():
 out=[]
 for k in range(1,13):
  s=(ROOT/'data'/f'loto6-chunk-{k}.js').read_text(encoding='utf-8')
  payload=s.split('push(',1)[1].rsplit(');',1)[0]
  for r in json.loads(payload):
   out.append((int(r[0]),tuple(int(x) for x in r[2:8]),int(r[8])))
 out.sort(key=lambda x:x[0]);return out

def z(x):
 x=np.asarray(x,float);return (x-x.mean())/(x.std()+1e-9)
def band(a):
 a=np.asarray(a);return (int(np.sum(a<=9)),int(np.sum((a>=10)&(a<=19))),int(np.sum((a>=20)&(a<=29))),int(np.sum((a>=30)&(a<=39))),int(np.sum(a>=40)))
def bcode(b):return b[0]*2401+b[1]*343+b[2]*49+b[3]*7+b[4]
def support_hist(agents,C,topn=500):
 weighted=np.zeros(44,float);raw=np.zeros(44,int);coverage=np.zeros(44,int)
 for _,idx in agents.items():
  seen=set()
  for rank,i in enumerate(idx[:topn],1):
   w=1/np.log2(rank+2.0)
   for x in map(int,C[int(i)]):weighted[x]+=w;raw[x]+=1;seen.add(x)
  for x in seen:coverage[x]+=1
 return weighted,raw,coverage

def direct_features(combs,qcuts):
 C=np.asarray(combs,np.int16);s=C.sum(1);sb=np.clip((s-21)//5,0,50).astype(np.int16);odd=(C%2).sum(1).astype(np.int8);con=(np.diff(C,axis=1)==1).sum(1).astype(np.int8)
 bands=np.array([bcode(band(r)) for r in C],np.int32);gap=np.digitize(np.std(np.diff(C,axis=1),axis=1),qcuts).astype(np.int8)
 return C,{'sum':sb,'odd':odd,'band':bands,'consec':con,'gap':gap}
def direct_core(C,pairc):
 total=np.zeros(len(C),np.float32);inc=np.zeros((len(C),6),np.float32)
 for i,j in itertools.combinations(range(6),2):
  v=pairc[C[:,i],C[:,j]].astype(np.float32);total+=v;inc[:,i]+=v;inc[:,j]+=v
 c5=np.max(total[:,None]-inc,axis=1);c4=np.full(len(C),-1e9,np.float32)
 for i,j in itertools.combinations(range(6),2):c4=np.maximum(c4,total-inc[:,i]-inc[:,j]+pairc[C[:,i],C[:,j]])
 return c5,c4

def sum_bin(s):
 if s<=115:return 'L<=115'
 if s<=135:return 'M116-135'
 if s<=155:return 'C136-155'
 if s<=175:return 'H156-175'
 return 'X>=176'

def choose_diverse(order,combs,meta,n=10):
 quota={'L<=115':2,'M116-135':2,'C136-155':3,'H156-175':2,'X>=176':1}
 usedbin=Counter();shape_count=Counter();num_count=Counter();pair_count=Counter();triple_count=Counter();sel=[]
 for i in order:
  row=tuple(combs[int(i)]);sh=meta[int(i)]['shape'];sb=sum_bin(sum(row))
  if usedbin[sb]>=quota[sb] or shape_count[sh]>=2:continue
  pairs=list(itertools.combinations(row,2));triples=list(itertools.combinations(row,3))
  if any(pair_count[x]>=2 for x in pairs) or any(triple_count[x]>=1 for x in triples) or any(num_count[x]>=4 for x in row):continue
  sel.append(int(i));usedbin[sb]+=1;shape_count[sh]+=1
  for x in row:num_count[x]+=1
  for x in pairs:pair_count[x]+=1
  for x in triples:triple_count[x]+=1
  if len(sel)==n:break
 if len(sel)<n:
  for i in order:
   if int(i) in sel:continue
   sh=meta[int(i)]['shape']
   if shape_count[sh]>=2:continue
   sel.append(int(i));shape_count[sh]+=1
   if len(sel)==n:break
 return sel

def main():
 rows=local_rows();rows=[r for r in rows if r[0]<=2134];assert rows[-1][0]==2134
 draws=np.asarray([r[1] for r in rows],np.int16);bonus=np.asarray([r[2] for r in rows],np.int16);t=len(draws)
 C=p.fixed_sample();st,inc,qcuts=p.build_static(C);draws2,bonus2,npref,ppref,actual=p.hist_actual(rows,qcuts);sizes,priors=p.prepare_priors(st)
 ss={}
 for h in (200,500,800):
  W=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W,st,inc,draws2,bonus2,npref)
 pairc=ppref[t]-ppref[max(0,t-300)];c5,c4=p.cores(C,pairc);comm=z(ss[500])+.20*z(c5)+.15*z(c4)
 agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
 weighted,raw,cov=support_hist(agents,C,500)
 ranknums=sorted([x for x in range(1,44) if x!=PREV_BONUS_EXCLUDE],key=lambda x:(weighted[x],raw[x],-x),reverse=True)
 recent20=set(map(int,draws[-20:].ravel()))
 core_candidates=[x for x in ranknums if x in recent20]
 core22=core_candidates[:22]
 satellite=[x for x in ranknums if x not in core22][:6]
 pool=sorted(set(core22+satellite))
 hist=[{'rank':i+1,'num':x,'weighted_support':float(weighted[x]),'raw_top500_occurrences':int(raw[x]),'agent_coverage':int(cov[x]),'recent20':x in recent20,'core22':x in core22,'satellite':x in satellite} for i,x in enumerate(ranknums)]
 sh20=Counter(band(r) for r in draws[-20:]);sh21_50=Counter(band(r) for r in draws[-50:-20]);sh50=Counter(band(r) for r in draws[-50:]);shall=Counter(band(r) for r in draws)
 prev=set(map(int,draws[-1]));core=set(core22);sat=set(satellite)
 combs=[];meta=[]
 for c in itertools.combinations(pool,6):
  if PREV_BONUS_EXCLUDE in c:continue
  ncore=sum(x in core for x in c);nsat=sum(x in sat for x in c)
  if ncore<4 or nsat>2:continue
  sh=band(c)
  if sh20[sh]>0:continue
  mode='21-50_once' if sh21_50[sh]==1 else ('new50' if sh50[sh]==0 else None)
  if mode is None:continue
  if len(set(c)&prev)>1:continue
  combs.append(c);meta.append({'shape':sh,'mode':mode,'ncore':ncore,'nsat':nsat})
 X,fx=direct_features(combs,qcuts)
 W=p.weights(t,500,actual,sizes,priors);pf=np.zeros(44,np.int8);pf[draws[-1]]=1;p2f=np.zeros(44,np.int8);p2f[draws[-2]]=1;c300=npref[t]-npref[max(0,t-300)];hot=(np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15];hf=np.zeros(44,np.int8);hf[hot]=1
 po=pf[X].sum(1);p2=p2f[X].sum(1);pb=(X==int(bonus[-1])).any(1).astype(np.int8);hh=hf[X].sum(1)
 stat=(W['sum'][fx['sum']]+W['odd'][fx['odd']]+W['band'][fx['band']]+W['consec'][fx['consec']]+W['gap'][fx['gap']]+W['prev'][po]+W['prev2'][p2]+W['pbonus'][pb]+W['hot'][hh]).astype(np.float32)
 dc5,dc4=direct_core(X,pairc);score=(stat-ss[500].mean())/(ss[500].std()+1e-9)+.20*(dc5-c5.mean())/(c5.std()+1e-9)+.15*(dc4-c4.mean())/(c4.std()+1e-9)
 order=np.argsort(score)[::-1];sel=choose_diverse(order,combs,meta,10)
 tickets=[]
 for i in sel:
  m=meta[i];tickets.append({'nums':list(combs[i]),'sum':sum(combs[i]),'sum_bin':sum_bin(sum(combs[i])),'shape':list(m['shape']),'shape_long_freq':shall[m['shape']]/t,'shape_21_50_count':sh21_50[m['shape']],'shape_50_count':sh50[m['shape']],'mode':m['mode'],'core_count':m['ncore'],'satellite_count':m['nsat'],'prev_overlap':len(set(combs[i])&prev),'stat_score':float(stat[i]),'core5':float(dc5[i]),'core4':float(dc4[i]),'committee_score':float(score[i]),'rebuild_rank':int(np.where(order==i)[0][0])+1})
 out={'target_draw':2135,'history_last':2134,'method':'Stat200/500/800 + Committee Top500 support -> recent20 Core22 + Satellite6 -> temporal band rebuild -> sum/shape diversified Committee rerank','formula':'Z(Stat500)+0.20Z(5core)+0.15Z(4core), z ref=fixed 60k pre-draw sample','previous_bonus_excluded':PREV_BONUS_EXCLUDE,'core22':core22,'satellite6':satellite,'pool28':pool,'histogram':hist,'rebuild_candidate_count':len(combs),'tickets':tickets}
 (OUT/'loto6_hist_rebuild_2135.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
