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

TICKETS=[
(2,7,16,18,37,42),(5,8,16,18,35,41),(2,5,26,34,35,37),(2,7,28,30,35,36),(5,7,26,32,33,36),
(8,16,26,28,29,42),(7,8,29,32,35,37),(5,8,28,34,36,38),(2,16,26,30,41,43),(5,18,28,30,40,42)]
POOL21=(2,5,7,8,16,18,26,28,29,30,32,33,34,35,36,37,38,40,41,42,43)
EXCLUDE={9,10,12,14,17,19,21,23,24,27,31,39}
SUM_BUCKETS=[(120,129),(130,139),(140,149),(150,159),(160,169)]

def zref(v,ref): return (float(v)-float(ref.mean()))/(float(ref.std())+1e-9)
def band(a):
 a=np.asarray(a);return (int(np.sum(a<=9)),int(np.sum((a>=10)&(a<=19))),int(np.sum((a>=20)&(a<=29))),int(np.sum((a>=30)&(a<=39))),int(np.sum(a>=40)))
def bcode(b): return b[0]*2401+b[1]*343+b[2]*49+b[3]*7+b[4]
def direct_static(rows,qcuts):
 C=np.asarray(rows,np.int16);s=C.sum(1);sb=np.clip((s-21)//5,0,50).astype(np.int16);odd=(C%2).sum(1).astype(np.int8);con=(np.diff(C,axis=1)==1).sum(1).astype(np.int8)
 bands=np.array([bcode(band(r)) for r in C],np.int32);gap=np.digitize(np.std(np.diff(C,axis=1),axis=1),qcuts).astype(np.int8)
 return C,{'sum':sb,'odd':odd,'band':bands,'consec':con,'gap':gap}
def direct_score(t,rowsC,fx,W,draws,bonus,npref,pairc,stat_ref,c5_ref,c4_ref):
 C=np.asarray(rowsC,np.int16);pf=np.zeros(44,np.int8);pf[draws[t-1]]=1;p2f=np.zeros(44,np.int8);p2f[draws[t-2]]=1;c300=npref[t]-npref[max(0,t-300)];hot=(np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15];hf=np.zeros(44,np.int8);hf[hot]=1
 po=pf[C].sum(1);p2= p2f[C].sum(1);pb=(C==int(bonus[t-1])).any(1).astype(np.int8);hh=hf[C].sum(1)
 st=(W['sum'][fx['sum']]+W['odd'][fx['odd']]+W['band'][fx['band']]+W['consec'][fx['consec']]+W['gap'][fx['gap']]+W['prev'][po]+W['prev2'][p2]+W['pbonus'][pb]+W['hot'][hh]).astype(np.float32)
 total=np.zeros(len(C),np.float32);inc=np.zeros((len(C),6),np.float32)
 for i,j in itertools.combinations(range(6),2):
  v=pairc[C[:,i],C[:,j]].astype(np.float32);total+=v;inc[:,i]+=v;inc[:,j]+=v
 c5=np.max(total[:,None]-inc,axis=1);c4=np.full(len(C),-1e9,np.float32)
 for i,j in itertools.combinations(range(6),2):c4=np.maximum(c4,total-inc[:,i]-inc[:,j]+pairc[C[:,i],C[:,j]])
 comm=(st-stat_ref.mean())/(stat_ref.std()+1e-9)+.20*(c5-c5_ref.mean())/(c5_ref.std()+1e-9)+.15*(c4-c4_ref.mean())/(c4_ref.std()+1e-9)
 return st,c5,c4,comm

def main():
 rows=p.fetch_history();rows=[r for r in rows if r[0]<=2133];di={d:i for i,(d,_,_) in enumerate(rows)};draws=np.asarray([r[1] for r in rows],np.int16);bonus=np.asarray([r[2] for r in rows],np.int16);t=len(draws)
 assert rows[-1][0]==2133
 Cref=p.fixed_sample();stref,inc,qcuts=p.build_static(Cref);draws2,bonus2,npref,ppref,actual=p.hist_actual(rows,qcuts);sizes,priors=p.prepare_priors(stref)
 # current refs and scores
 W=p.weights(t,500,actual,sizes,priors);sref=p.stat_score(t,500,W,stref,inc,draws,bonus,npref);pairc=ppref[t]-ppref[max(0,t-300)];c5ref,c4ref=p.cores(Cref,pairc);commref=(sref-sref.mean())/(sref.std()+1e-9)+.20*(c5ref-c5ref.mean())/(c5ref.std()+1e-9)+.15*(c4ref-c4ref.mean())/(c4ref.std()+1e-9)
 X,fx=direct_static(TICKETS,qcuts);st,c5,c4,comm=direct_score(t,X,fx,W,draws,bonus,npref,pairc,sref,c5ref,c4ref)
 sh20=Counter(band(r) for r in draws[-20:]);sh2150=Counter(band(r) for r in draws[-50:-20]);sh50=Counter(band(r) for r in draws[-50:]);recentnums=set(map(int,draws[-20:].ravel()));prev=set(map(int,draws[-1]))
 # restricted pool candidates after final conditions
 valid=[];vmeta=[]
 for c in itertools.combinations(POOL21,6):
  sh=band(c);temporal=(sh20[sh]==0 and (sh2150[sh]==1 or sh50[sh]==0))
  if not temporal or len(set(c)&prev)>1:continue
  valid.append(c);vmeta.append(sh)
 VX,vfx=direct_static(valid,qcuts);vst,vc5,vc4,vcomm=direct_score(t,VX,vfx,W,draws,bonus,npref,pairc,sref,c5ref,c4ref);vorder=np.argsort(vcomm)[::-1];vrank=np.empty(len(valid),int);vrank[vorder]=np.arange(1,len(valid)+1);vdict={tuple(c):i for i,c in enumerate(valid)}
 audit=[]
 for i,c in enumerate(TICKETS):
  sh=band(c);temporal=(sh20[sh]==0 and (sh2150[sh]==1 or sh50[sh]==0));all_recent=all(x in recentnums for x in c);excluded=sorted(set(c)&EXCLUDE);inpool=all(x in POOL21 for x in c);po=len(set(c)&prev);vi=vdict.get(tuple(c));pct=float((np.sum(commref>comm[i])+1)/(len(commref)+1)*100)
  audit.append({'nums':list(c),'sum':sum(c),'shape':list(sh),'all_nums_seen_prior20':all_recent,'excluded_present':excluded,'in_pool21':inpool,'shape_prior20_count':sh20[sh],'shape_prior21_50_count':sh2150[sh],'shape_prior50_count':sh50[sh],'temporal_ok':temporal,'prev_overlap':po,'prev_overlap_ok':po<=1,'all_rules_ok':bool(all_recent and not excluded and inpool and temporal and po<=1),'stat_score':float(st[i]),'core5':float(c5[i]),'core4':float(c4[i]),'committee_score':float(comm[i]),'approx_top_percent_vs_60k_ref':pct,'rank_within_valid_rebuild':int(vrank[vi]) if vi is not None else None,'valid_rebuild_count':len(valid)})
 # historical walk-forward benchmark 1628-2133, same structural state; current exclusion list deliberately NOT applied historically
 hist=[]
 for draw in range(1628,2134):
  ti=di.get(draw)
  if ti is None or ti<50:continue
  win=tuple(map(int,draws[ti]));sh=band(win);p20=Counter(band(r) for r in draws[max(0,ti-20):ti]);p2150=Counter(band(r) for r in draws[max(0,ti-50):max(0,ti-20)]);p50=Counter(band(r) for r in draws[max(0,ti-50):ti]);temporal=(p20[sh]==0 and (p2150[sh]==1 or p50[sh]==0));all_recent=all(x in set(map(int,draws[max(0,ti-20):ti].ravel())) for x in win)
  if not temporal:continue
  Wi=p.weights(ti,500,actual,sizes,priors);sr=p.stat_score(ti,500,Wi,stref,inc,draws,bonus,npref);pc=ppref[ti]-ppref[max(0,ti-300)];r5,r4=p.cores(Cref,pc);cr=(sr-sr.mean())/(sr.std()+1e-9)+.20*(r5-r5.mean())/(r5.std()+1e-9)+.15*(r4-r4.mean())/(r4.std()+1e-9)
  WX,wfx=direct_static([win],qcuts);ws,w5,w4,wc=direct_score(ti,WX,wfx,Wi,draws,bonus,npref,pc,sr,r5,r4);top=float((np.sum(cr>wc[0])+1)/(len(cr)+1)*100)
  hist.append({'draw':draw,'winner':list(win),'sum':sum(win),'shape':list(sh),'all_winner_nums_seen_prior20':all_recent,'committee_score':float(wc[0]),'approx_top_percent_vs_60k_ref':top})
 def summ(xs):
  a=np.array(xs,float)
  return {'n':len(a),'min':float(a.min()) if len(a) else None,'q25':float(np.quantile(a,.25)) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'mean':float(a.mean()) if len(a) else None,'q75':float(np.quantile(a,.75)) if len(a) else None,'max':float(a.max()) if len(a) else None}
 hist_strict=[h for h in hist if h['all_winner_nums_seen_prior20']]
 out={'target_draw':2134,'score_formula':'Z(Stat500)+0.20Z(5core)+0.15Z(4core), Z reference=fixed pre-draw 60k sample','final_pool21':list(POOL21),'final_exclusions':sorted(EXCLUDE),'current_ticket_audit':audit,'current_valid_rebuild_count':len(valid),'historical_definition':'winner shape absent prior20 AND (shape appears exactly once in prior21-50 OR absent prior50); current 2134-specific exclusion list is not applied historically','historical_same_shape_cases':{'score_summary':summ([h['committee_score'] for h in hist]),'top_percent_summary':summ([h['approx_top_percent_vs_60k_ref'] for h in hist]),'last15':hist[-15:]},'historical_strict_plus_all_winner_nums_seen_prior20':{'score_summary':summ([h['committee_score'] for h in hist_strict]),'top_percent_summary':summ([h['approx_top_percent_vs_60k_ref'] for h in hist_strict]),'last15':hist_strict[-15:]}}
 (OUT/'loto6_rebuild_2134_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
