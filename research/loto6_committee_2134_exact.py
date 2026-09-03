#!/usr/bin/env python3
from __future__ import annotations
import csv,io,itertools,json,math,urllib.request
from collections import Counter
from pathlib import Path
import numpy as np

URL='https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv'
NALL=math.comb(43,6); ALPHA=75.; CLIP=1.5
MANUAL=[
(8,18,20,25,28,43),(8,16,18,19,28,36),(15,18,23,25,28,36),(5,8,25,32,34,36),(19,20,25,28,36,42),
(1,16,25,36,40,42),(2,3,6,19,20,37),(3,6,16,25,26,37),(3,6,16,19,30,40),(2,13,17,26,29,40)]

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
def band_code_tuple(b):return b[0]*2401+b[1]*343+b[2]*49+b[3]*7+b[4]
def hyp(K):
 den=math.comb(43,6);a=np.zeros(7,float)
 for k in range(7):
  if k<=K and 6-k<=43-K:a[k]=math.comb(K,k)*math.comb(43-K,6-k)/den
 return a

def diverse(ids,score,C,n=10):
 order=ids[np.argsort(score[ids])[::-1]][:1500];sel=[];tc=Counter();pc=Counter();nc=Counter()
 for i in order:
  row=tuple(map(int,C[int(i)]));trs=list(itertools.combinations(row,3));pas=list(itertools.combinations(row,2))
  if any(tc[x]>=1 for x in trs) or any(pc[x]>=2 for x in pas) or any(nc[x]>=4 for x in row):continue
  sel.append(int(i));
  for x in trs:tc[x]+=1
  for x in pas:pc[x]+=1
  for x in row:nc[x]+=1
  if len(sel)==n:break
 if len(sel)<n:
  used=set(sel)
  for i in order:
   if int(i) not in used:sel.append(int(i));used.add(int(i))
   if len(sel)==n:break
 return sel

def rank_of(arr,v):return int(np.count_nonzero(arr>v)+1)

def main():
 rows=fetch_history(); assert rows[-1][0]>=2133
 rows=[r for r in rows if r[0]<=2133]; draws=np.asarray([r[1] for r in rows],np.int16); bonus=np.asarray([r[2] for r in rows],np.int16);T=len(draws);t=T
 print('history',rows[-1],T,flush=True)
 print('generate all',NALL,flush=True)
 C=np.fromiter((x for c in itertools.combinations(range(1,44),6) for x in c),dtype=np.int16,count=NALL*6).reshape(NALL,6)
 sums=C.sum(1);sb=np.clip((sums-21)//5,0,50).astype(np.int16);odd=(C%2).sum(1).astype(np.int8);con=(np.diff(C,axis=1)==1).sum(1).astype(np.int8)
 b0=(C<=9).sum(1);b1=((C>=10)&(C<=19)).sum(1);b2=((C>=20)&(C<=29)).sum(1);b3=((C>=30)&(C<=39)).sum(1);b4=(C>=40).sum(1)
 band=(b0*2401+b1*343+b2*49+b3*7+b4).astype(np.int32);gs=np.std(np.diff(C,axis=1),axis=1);qcuts=np.quantile(gs,[.2,.4,.6,.8]);gap=np.digitize(gs,qcuts).astype(np.int8)
 sizes={'sum':51,'odd':7,'band':16807,'consec':6,'gap':5,'prev':7,'prev2':7,'pbonus':2,'hot':7}
 priors={
 'sum':np.bincount(sb,minlength=51).astype(float),'odd':np.bincount(odd,minlength=7).astype(float),'band':np.bincount(band,minlength=16807).astype(float),'consec':np.bincount(con,minlength=6).astype(float),'gap':np.bincount(gap,minlength=5).astype(float)}
 for k in priors:priors[k]/=priors[k].sum()
 priors['prev']=hyp(6);priors['prev2']=hyp(6);priors['hot']=hyp(15);priors['pbonus']=np.array([37/43,6/43],float)
 # rolling counts and actual categories; qcuts are frozen from candidate universe
 npref=np.zeros((T+1,44),np.int32);ppref=np.zeros((T+1,44,44),np.int16)
 for u,row in enumerate(draws):
  npref[u+1]=npref[u];npref[u+1,row]+=1;ppref[u+1]=ppref[u]
  for a,b in itertools.combinations(map(int,row),2):ppref[u+1,a,b]+=1;ppref[u+1,b,a]+=1
 actual=[]
 for u,row in enumerate(draws):
  bt=band_tuple(row);cur=set(map(int,row));prev=set(map(int,draws[u-1])) if u else set();prev2=set(map(int,draws[u-2])) if u>=2 else set();c300=npref[u]-npref[max(0,u-300)];hot=set((np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]) if u else set()
  actual.append({'sum':int(np.clip((int(row.sum())-21)//5,0,50)),'odd':int((row%2).sum()),'band':band_code_tuple(bt),'consec':int((np.diff(row)==1).sum()),'gap':int(np.digitize(np.std(np.diff(row)),qcuts)),'prev':len(cur&prev),'prev2':len(cur&prev2),'pbonus':int(u>=1 and int(bonus[u-1]) in cur),'hot':sum(int(x) in hot for x in row)})
 lo=max(0,t-500);W={}
 for f in ('sum','odd','band','consec','gap','prev','prev2','pbonus','hot'):
  wins=np.bincount([actual[u][f] for u in range(lo,t)],minlength=sizes[f]).astype(float);p=priors[f];q=(wins+ALPHA*p)/(t-lo+ALPHA);W[f]=np.clip(np.log(np.maximum(q,1e-15)/np.maximum(p,1e-15)),-CLIP,CLIP)
 prev=draws[-1];prev2=draws[-2];c300=npref[t]-npref[max(0,t-300)];hot=(np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]
 pf=np.zeros(44,np.int8);pf[prev]=1;p2f=np.zeros(44,np.int8);p2f[prev2]=1;hf=np.zeros(44,np.int8);hf[hot]=1
 po=pf[C].sum(1);p2=p2f[C].sum(1);hh=hf[C].sum(1);pb=(C==int(bonus[-1])).any(1).astype(np.int8)
 stat=(W['sum'][sb]+W['odd'][odd]+W['band'][band]+W['consec'][con]+W['gap'][gap]+W['prev'][po]+W['prev2'][p2]+W['pbonus'][pb]+W['hot'][hh]).astype(np.float32)
 # gate used only for conventional Main diversified portfolio
 low=(C<=22).sum(1);gate=((sums>=125)&(sums<=145)).astype(np.int8)+((odd>=2)&(odd<=4)).astype(np.int8)+((low>=2)&(low<=4)).astype(np.int8)+(b2>=1).astype(np.int8)+((C>=32).sum(1)>=1).astype(np.int8)+(po<=1).astype(np.int8)
 suffix=np.zeros(10,np.int8);suffix[prev%10]=1;so=suffix[C%10].sum(1);gate+=( (so>=2)&(so<=4) ).astype(np.int8)+(pb==0).astype(np.int8)
 pairc=ppref[t]-ppref[max(0,t-300)];c5=np.empty(NALL,np.float32);c4=np.empty(NALL,np.float32)
 CH=350000
 for a in range(0,NALL,CH):
  z=min(NALL,a+CH);X=C[a:z];total=np.zeros(z-a,np.float32);inc=np.zeros((z-a,6),np.float32)
  for i,j in itertools.combinations(range(6),2):
   v=pairc[X[:,i],X[:,j]].astype(np.float32);total+=v;inc[:,i]+=v;inc[:,j]+=v
  c5[a:z]=np.max(total[:,None]-inc,axis=1);v4=np.full(z-a,-1e9,np.float32)
  for i,j in itertools.combinations(range(6),2):v4=np.maximum(v4,total-inc[:,i]-inc[:,j]+pairc[X[:,i],X[:,j]])
  c4[a:z]=v4
  if a%(CH*5)==0:print('core',a,'/',NALL,flush=True)
 zs=(stat-stat.mean())/(stat.std()+1e-9);z5=(c5-c5.mean())/(c5.std()+1e-9);z4=(c4-c4.mean())/(c4.std()+1e-9);comm=(zs+.20*z5+.15*z4).astype(np.float32)
 # pure Committee top10 + conventional gate6 diversified top10
 raw_idx=np.argpartition(comm,-10)[-10:];raw_idx=raw_idx[np.argsort(comm[raw_idx])[::-1]]
 gateids=np.where(gate>=6)[0];M=min(5000,len(gateids));gi=gateids[np.argpartition(comm[gateids],-M)[-M:]];main_sel=diverse(gi,comm,C,10)
 shapes=Counter(band_tuple(r) for r in draws)
 def rec(i):
  row=tuple(map(int,C[int(i)]));sh=band_tuple(row);return {'nums':row,'sum':int(sum(row)),'shape':sh,'shape_freq':shapes[sh]/T,'stat_score':float(stat[i]),'stat_rank':rank_of(stat,stat[i]),'core5':float(c5[i]),'core5_rank':rank_of(c5,c5[i]),'core4':float(c4[i]),'core4_rank':rank_of(c4,c4[i]),'committee':float(comm[i]),'committee_rank':rank_of(comm,comm[i]),'gate':int(gate[i])}
 raw=[rec(i) for i in raw_idx];main=[rec(i) for i in main_sel]
 # exact locate each manual candidate by combinadic lookup using dictionary only for 10 rows
 key_to_i={tuple(map(int,C[i])):i for i in range(NALL) if False}
 # binary lex search via structured void view
 dt=np.dtype((np.void,C.dtype.itemsize*6));Cv=np.ascontiguousarray(C).view(dt).ravel()
 manual=[]
 for row in MANUAL:
  rv=np.asarray(row,dtype=np.int16).reshape(1,6).view(dt).ravel()[0];i=int(np.searchsorted(Cv,rv));assert tuple(map(int,C[i]))==row;manual.append(rec(i))
 out={'target_draw':2134,'history_last':rows[-1][0],'normalization':'exact all-6,096,454 population z for Stat/5core/4core','formula':'Z(Stat500)+0.20*Z(5core)+0.15*Z(4core)','raw_committee_top10':raw,'main_gate6_diversified10':main,'manual10_ranks':manual,'hot15':list(map(int,hot)),'previous':list(map(int,prev)),'previous_bonus':int(bonus[-1])}
 Path('research/results').mkdir(parents=True,exist_ok=True);Path('research/results/loto6_committee_2134_exact.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
