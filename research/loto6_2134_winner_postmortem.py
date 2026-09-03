#!/usr/bin/env python3
from __future__ import annotations
import csv,io,itertools,json,math,urllib.request
from pathlib import Path
import numpy as np

URL='https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv'
WIN=(5,9,10,19,26,35)
NALL=math.comb(43,6); ALPHA=75.; CLIP=1.5

def fetch_history():
 raw=urllib.request.urlopen(URL,timeout=60).read()
 try:text=raw.decode('cp932')
 except UnicodeDecodeError:text=raw.decode('utf-8')
 rd=csv.reader(io.StringIO(text));next(rd);rows=[]
 for r in rd:
  try: rows.append((int(r[0]),tuple(sorted(int(r[2+i]) for i in range(6))),int(r[8])))
  except: pass
 rows.sort();return [r for r in rows if r[0]<=2133]

def band_tuple(a):
 a=np.asarray(a);return (int((a<=9).sum()),int(((a>=10)&(a<=19)).sum()),int(((a>=20)&(a<=29)).sum()),int(((a>=30)&(a<=39)).sum()),int((a>=40).sum()))
def bcode(b):return b[0]*2401+b[1]*343+b[2]*49+b[3]*7+b[4]
def hyp(K):
 den=math.comb(43,6);a=np.zeros(7,float)
 for k in range(7):
  if k<=K and 6-k<=43-K:a[k]=math.comb(K,k)*math.comb(43-K,6-k)/den
 return a

def main():
 rows=fetch_history();draws=np.asarray([r[1] for r in rows],np.int16);bonus=np.asarray([r[2] for r in rows],np.int16);T=len(draws);t=T
 C=np.fromiter((x for c in itertools.combinations(range(1,44),6) for x in c),dtype=np.int16,count=NALL*6).reshape(NALL,6)
 sums=C.sum(1);sb=np.clip((sums-21)//5,0,50).astype(np.int16);odd=(C%2).sum(1).astype(np.int8);con=(np.diff(C,axis=1)==1).sum(1).astype(np.int8)
 b0=(C<=9).sum(1);b1=((C>=10)&(C<=19)).sum(1);b2=((C>=20)&(C<=29)).sum(1);b3=((C>=30)&(C<=39)).sum(1);b4=(C>=40).sum(1);band=(b0*2401+b1*343+b2*49+b3*7+b4).astype(np.int32)
 gs=np.std(np.diff(C,axis=1),axis=1);qcuts=np.quantile(gs,[.2,.4,.6,.8]);gap=np.digitize(gs,qcuts).astype(np.int8)
 sizes={'sum':51,'odd':7,'band':16807,'consec':6,'gap':5,'prev':7,'prev2':7,'pbonus':2,'hot':7}
 priors={'sum':np.bincount(sb,minlength=51).astype(float),'odd':np.bincount(odd,minlength=7).astype(float),'band':np.bincount(band,minlength=16807).astype(float),'consec':np.bincount(con,minlength=6).astype(float),'gap':np.bincount(gap,minlength=5).astype(float)}
 for k in priors:priors[k]/=priors[k].sum()
 priors['prev']=hyp(6);priors['prev2']=hyp(6);priors['hot']=hyp(15);priors['pbonus']=np.array([37/43,6/43],float)
 npref=np.zeros((T+1,44),np.int32);ppref=np.zeros((T+1,44,44),np.int16)
 for u,row in enumerate(draws):
  npref[u+1]=npref[u];npref[u+1,row]+=1;ppref[u+1]=ppref[u]
  for a,b in itertools.combinations(map(int,row),2):ppref[u+1,a,b]+=1;ppref[u+1,b,a]+=1
 actual=[]
 for u,row in enumerate(draws):
  cur=set(map(int,row));prev=set(map(int,draws[u-1])) if u else set();prev2=set(map(int,draws[u-2])) if u>=2 else set();c300=npref[u]-npref[max(0,u-300)];hot=set((np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]) if u else set()
  actual.append({'sum':int(np.clip((int(row.sum())-21)//5,0,50)),'odd':int((row%2).sum()),'band':bcode(band_tuple(row)),'consec':int((np.diff(row)==1).sum()),'gap':int(np.digitize(np.std(np.diff(row)),qcuts)),'prev':len(cur&prev),'prev2':len(cur&prev2),'pbonus':int(u>=1 and int(bonus[u-1]) in cur),'hot':sum(int(x) in hot for x in row)})
 lo=max(0,t-500);W={}
 for f in ('sum','odd','band','consec','gap','prev','prev2','pbonus','hot'):
  wins=np.bincount([actual[u][f] for u in range(lo,t)],minlength=sizes[f]).astype(float);p=priors[f];q=(wins+ALPHA*p)/(t-lo+ALPHA);W[f]=np.clip(np.log(np.maximum(q,1e-15)/np.maximum(p,1e-15)),-CLIP,CLIP)
 prev=draws[-1];prev2=draws[-2];c300=npref[t]-npref[max(0,t-300)];hot=(np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]
 pf=np.zeros(44,np.int8);pf[prev]=1;p2f=np.zeros(44,np.int8);p2f[prev2]=1;hf=np.zeros(44,np.int8);hf[hot]=1
 po=pf[C].sum(1);p2=p2f[C].sum(1);hh=hf[C].sum(1);pb=(C==int(bonus[-1])).any(1).astype(np.int8)
 stat=(W['sum'][sb]+W['odd'][odd]+W['band'][band]+W['consec'][con]+W['gap'][gap]+W['prev'][po]+W['prev2'][p2]+W['pbonus'][pb]+W['hot'][hh]).astype(np.float32)
 pairc=ppref[t]-ppref[max(0,t-300)];c5=np.empty(NALL,np.float32);c4=np.empty(NALL,np.float32);CH=350000
 for a in range(0,NALL,CH):
  z=min(NALL,a+CH);X=C[a:z];total=np.zeros(z-a,np.float32);inc=np.zeros((z-a,6),np.float32)
  for i,j in itertools.combinations(range(6),2):
   v=pairc[X[:,i],X[:,j]].astype(np.float32);total+=v;inc[:,i]+=v;inc[:,j]+=v
  c5[a:z]=np.max(total[:,None]-inc,axis=1);v4=np.full(z-a,-1e9,np.float32)
  for i,j in itertools.combinations(range(6),2):v4=np.maximum(v4,total-inc[:,i]-inc[:,j]+pairc[X[:,i],X[:,j]])
  c4[a:z]=v4
 comm=((stat-stat.mean())/(stat.std()+1e-9)+.20*(c5-c5.mean())/(c5.std()+1e-9)+.15*(c4-c4.mean())/(c4.std()+1e-9)).astype(np.float32)
 # locate winner lexicographically
 dt=np.dtype((np.void,C.dtype.itemsize*6));Cv=np.ascontiguousarray(C).view(dt).ravel();rv=np.asarray(WIN,dtype=np.int16).reshape(1,6).view(dt).ravel()[0];i=int(np.searchsorted(Cv,rv));assert tuple(map(int,C[i]))==WIN
 def rank(arr,v):return int(np.count_nonzero(arr>v)+1)
 sh=band_tuple(WIN)
 # temporal shape history
 sh20=[band_tuple(r) for r in draws[-20:]];sh2150=[band_tuple(r) for r in draws[-50:-20]];sh50=[band_tuple(r) for r in draws[-50:]]
 intervals={}
 for x in WIN:
  prev_idxs=np.where((draws==x).any(axis=1))[0]
  intervals[str(x)]=int(t-prev_idxs[-1]) if len(prev_idxs) else None
 out={'draw':2134,'winner':list(WIN),'sum':sum(WIN),'shape':list(sh),'odd_even':[sum(x%2 for x in WIN),sum(x%2==0 for x in WIN)],'prev_main_overlap':len(set(WIN)&set(map(int,prev))),'prev_bonus_in_winner':int(bonus[-1] in WIN),'shape_prior20_count':sh20.count(sh),'shape_prior21_50_count':sh2150.count(sh),'shape_prior50_count':sh50.count(sh),'intervals_before_draw':intervals,'stat_score':float(stat[i]),'stat_rank':rank(stat,stat[i]),'core5':float(c5[i]),'core5_rank':rank(c5,c5[i]),'core4':float(c4[i]),'core4_rank':rank(c4,c4[i]),'committee_score':float(comm[i]),'committee_rank':rank(comm,comm[i]),'committee_top_percent':rank(comm,comm[i])/NALL*100,'NALL':NALL}
 Path('research/results').mkdir(parents=True,exist_ok=True);Path('research/results/loto6_2134_winner_postmortem.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
