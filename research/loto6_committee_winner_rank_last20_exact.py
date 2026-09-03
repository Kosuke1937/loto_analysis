#!/usr/bin/env python3
from __future__ import annotations
import csv,io,itertools,json,math,urllib.request
from pathlib import Path
import numpy as np
from collections import Counter

URL='https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv'
NALL=math.comb(43,6); ALPHA=75.; CLIP=1.5
TARGET_DRAWS=list(range(2114,2134))

def fetch_history():
 raw=urllib.request.urlopen(URL,timeout=60).read()
 try:text=raw.decode('cp932')
 except UnicodeDecodeError:text=raw.decode('utf-8')
 rd=csv.reader(io.StringIO(text));next(rd);rows=[]
 for r in rd:
  try: rows.append((int(r[0]),tuple(sorted(int(r[2+i]) for i in range(6))),int(r[8])))
  except: pass
 rows.sort();return rows

def band_tuple(a):
 a=np.asarray(a);return (int((a<=9).sum()),int(((a>=10)&(a<=19)).sum()),int(((a>=20)&(a<=29)).sum()),int(((a>=30)&(a<=39)).sum()),int((a>=40).sum()))
def bcode(b):return b[0]*2401+b[1]*343+b[2]*49+b[3]*7+b[4]
def hyp(K):
 den=math.comb(43,6);a=np.zeros(7,float)
 for k in range(7):
  if k<=K and 6-k<=43-K:a[k]=math.comb(K,k)*math.comb(43-K,6-k)/den
 return a

def actual_categories(draws,bonus,npref,qcuts):
 T=len(draws);out=[]
 for u,row in enumerate(draws):
  cur=set(map(int,row));prev=set(map(int,draws[u-1])) if u else set();prev2=set(map(int,draws[u-2])) if u>=2 else set();c300=npref[u]-npref[max(0,u-300)];hot=set((np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]) if u else set()
  out.append({'sum':int(np.clip((int(row.sum())-21)//5,0,50)),'odd':int((row%2).sum()),'band':bcode(band_tuple(row)),'consec':int((np.diff(row)==1).sum()),'gap':int(np.digitize(np.std(np.diff(row)),qcuts)),'prev':len(cur&prev),'prev2':len(cur&prev2),'pbonus':int(u>=1 and int(bonus[u-1]) in cur),'hot':sum(int(x) in hot for x in row)})
 return out

def score_weights(t,actual,priors,sizes):
 lo=max(0,t-500);W={}
 for f in ('sum','odd','band','consec','gap','prev','prev2','pbonus','hot'):
  wins=np.bincount([actual[u][f] for u in range(lo,t)],minlength=sizes[f]).astype(float);p=priors[f];q=(wins+ALPHA*p)/(t-lo+ALPHA);W[f]=np.clip(np.log(np.maximum(q,1e-15)/np.maximum(p,1e-15)),-CLIP,CLIP)
 return W

def direct_one(row,t,W,draws,bonus,npref,pairc,qcuts,means_stds):
 x=np.asarray(row,np.int16);sb=int(np.clip((x.sum()-21)//5,0,50));odd=int((x%2).sum());bc=bcode(band_tuple(x));con=int((np.diff(x)==1).sum());gap=int(np.digitize(np.std(np.diff(x)),qcuts));prev=set(map(int,draws[t-1]));prev2=set(map(int,draws[t-2]));po=sum(int(v) in prev for v in x);p2=sum(int(v) in prev2 for v in x);pb=int(int(bonus[t-1]) in set(map(int,x)));c300=npref[t]-npref[max(0,t-300)];hot=set((np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]);hh=sum(int(v) in hot for v in x)
 st=float(W['sum'][sb]+W['odd'][odd]+W['band'][bc]+W['consec'][con]+W['gap'][gap]+W['prev'][po]+W['prev2'][p2]+W['pbonus'][pb]+W['hot'][hh])
 total=0.;inc=np.zeros(6,float)
 for i,j in itertools.combinations(range(6),2):
  v=float(pairc[x[i],x[j]]);total+=v;inc[i]+=v;inc[j]+=v
 c5=float(np.max(total-inc));c4=-1e9
 for i,j in itertools.combinations(range(6),2): c4=max(c4,float(total-inc[i]-inc[j]+pairc[x[i],x[j]]))
 ms,ss,m5,s5,m4,s4=means_stds;comm=(st-ms)/(ss+1e-9)+.20*(c5-m5)/(s5+1e-9)+.15*(c4-m4)/(s4+1e-9)
 return st,c5,c4,float(comm)

def main():
 rows=fetch_history(); rows=[r for r in rows if r[0]<=2133]; idx={d:i for i,(d,_,_) in enumerate(rows)};draws=np.asarray([r[1] for r in rows],np.int16);bonus=np.asarray([r[2] for r in rows],np.int16);T=len(draws)
 print('history last',rows[-1],flush=True)
 C=np.fromiter((x for c in itertools.combinations(range(1,44),6) for x in c),dtype=np.int16,count=NALL*6).reshape(NALL,6)
 sums=C.sum(1);sb=np.clip((sums-21)//5,0,50).astype(np.int16);odd=(C%2).sum(1).astype(np.int8);con=(np.diff(C,axis=1)==1).sum(1).astype(np.int8);b0=(C<=9).sum(1);b1=((C>=10)&(C<=19)).sum(1);b2=((C>=20)&(C<=29)).sum(1);b3=((C>=30)&(C<=39)).sum(1);b4=(C>=40).sum(1);band=(b0*2401+b1*343+b2*49+b3*7+b4).astype(np.int32);gs=np.std(np.diff(C,axis=1),axis=1);qcuts=np.quantile(gs,[.2,.4,.6,.8]);gap=np.digitize(gs,qcuts).astype(np.int8)
 sizes={'sum':51,'odd':7,'band':16807,'consec':6,'gap':5,'prev':7,'prev2':7,'pbonus':2,'hot':7}
 priors={'sum':np.bincount(sb,minlength=51).astype(float),'odd':np.bincount(odd,minlength=7).astype(float),'band':np.bincount(band,minlength=16807).astype(float),'consec':np.bincount(con,minlength=6).astype(float),'gap':np.bincount(gap,minlength=5).astype(float)}
 for k in priors:priors[k]/=priors[k].sum()
 priors['prev']=hyp(6);priors['prev2']=hyp(6);priors['hot']=hyp(15);priors['pbonus']=np.array([37/43,6/43],float)
 npref=np.zeros((T+1,44),np.int32);ppref=np.zeros((T+1,44,44),np.int16)
 for u,row in enumerate(draws):
  npref[u+1]=npref[u];npref[u+1,row]+=1;ppref[u+1]=ppref[u]
  for a,b in itertools.combinations(map(int,row),2):ppref[u+1,a,b]+=1;ppref[u+1,b,a]+=1
 actual=actual_categories(draws,bonus,npref,qcuts)
 results=[];CH=350000
 for drawno in TARGET_DRAWS:
  t=idx[drawno]; W=score_weights(t,actual,priors,sizes);pairc=ppref[t]-ppref[max(0,t-300)];prev=draws[t-1];prev2=draws[t-2];c300=npref[t]-npref[max(0,t-300)];hot=(np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]
  pf=np.zeros(44,np.int8);pf[prev]=1;p2f=np.zeros(44,np.int8);p2f[prev2]=1;hf=np.zeros(44,np.int8);hf[hot]=1
  po=pf[C].sum(1);p2=p2f[C].sum(1);hh=hf[C].sum(1);pb=(C==int(bonus[t-1])).any(1).astype(np.int8)
  stat=(W['sum'][sb]+W['odd'][odd]+W['band'][band]+W['consec'][con]+W['gap'][gap]+W['prev'][po]+W['prev2'][p2]+W['pbonus'][pb]+W['hot'][hh]).astype(np.float32)
  c5=np.empty(NALL,np.float32);c4=np.empty(NALL,np.float32)
  for a in range(0,NALL,CH):
   z=min(NALL,a+CH);X=C[a:z];total=np.zeros(z-a,np.float32);inc=np.zeros((z-a,6),np.float32)
   for i,j in itertools.combinations(range(6),2):
    v=pairc[X[:,i],X[:,j]].astype(np.float32);total+=v;inc[:,i]+=v;inc[:,j]+=v
   c5[a:z]=np.max(total[:,None]-inc,axis=1);v4=np.full(z-a,-1e9,np.float32)
   for i,j in itertools.combinations(range(6),2):v4=np.maximum(v4,total-inc[:,i]-inc[:,j]+pairc[X[:,i],X[:,j]])
   c4[a:z]=v4
  ms,ss=float(stat.mean()),float(stat.std());m5,s5=float(c5.mean()),float(c5.std());m4,s4=float(c4.mean()),float(c4.std())
  comm=((stat-ms)/(ss+1e-9)+.20*(c5-m5)/(s5+1e-9)+.15*(c4-m4)/(s4+1e-9)).astype(np.float32)
  winner=tuple(map(int,draws[t]));dt=np.dtype((np.void,C.dtype.itemsize*6));Cv=np.ascontiguousarray(C).view(dt).ravel();rv=np.asarray(winner,dtype=np.int16).reshape(1,6).view(dt).ravel()[0];wi=int(np.searchsorted(Cv,rv));assert tuple(map(int,C[wi]))==winner
  wrank=int(np.count_nonzero(comm>comm[wi])+1);srank=int(np.count_nonzero(stat>stat[wi])+1);r5=int(np.count_nonzero(c5>c5[wi])+1);r4=int(np.count_nonzero(c4>c4[wi])+1)
  results.append({'draw':drawno,'winner':list(winner),'sum':sum(winner),'stat_score':float(stat[wi]),'stat_rank':srank,'core5':float(c5[wi]),'core5_rank':r5,'core4':float(c4[wi]),'core4_rank':r4,'committee_score':float(comm[wi]),'committee_rank':wrank,'committee_top_percent':100*wrank/NALL})
  print(drawno,winner,'comm',float(comm[wi]),'rank',wrank,flush=True)
 out={'formula':'Z_exact(Stat500)+0.20 Z_exact(5core300)+0.15 Z_exact(4core300)','universe':NALL,'draws':results}
 Path('research/results').mkdir(parents=True,exist_ok=True);Path('research/results/loto6_committee_winner_rank_last20_exact.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
