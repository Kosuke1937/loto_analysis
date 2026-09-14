#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)
BANDS=((1,9),(10,19),(20,29),(30,39),(40,43))
LOCAL=((1,5),(6,9),(10,14),(15,19),(20,24),(25,29),(30,34),(35,39),(40,43))
NALL=math.comb(43,6); ALPHA=75.; CLIP=1.5

def sum_state(s):
    if s<=109:return 'S1'
    if s<=129:return 'S2'
    if s<=149:return 'S3'
    return 'S4'

def band(row):
    return tuple(sum(lo<=x<=hi for x in row) for lo,hi in BANDS)

def band_code_tuple(b):return b[0]*2401+b[1]*343+b[2]*49+b[3]*7+b[4]
def layer(freq):return 'A' if freq>=.02 else ('B' if freq>=.01 else ('C' if freq>=.005 else 'D'))
def local_cluster(n):
    for i,(lo,hi) in enumerate(LOCAL):
        if lo<=n<=hi:return i
    raise ValueError(n)
def adj(nums):
    out=set()
    for x in nums:
        if x>1:out.add(x-1)
        if x<43:out.add(x+1)
    return out-set(nums)
def has_three_consecutive(row):
    return any(row[i+1]==row[i]+1 and row[i+2]==row[i]+2 for i in range(4))
def load_rows():
    rows=[]
    for k in range(1,13):
        p=ROOT/'data'/f'loto6-chunk-{k}.js'
        if not p.exists():continue
        s=p.read_text(encoding='utf-8'); payload=s.split('push(',1)[1].rsplit(');',1)[0]
        for r in json.loads(payload):rows.append((int(r[0]),tuple(int(x) for x in r[2:8]),int(r[8])))
    rows.sort();return rows

def flow_sets(hist):
    f1=set(hist[-1]);f2=adj(f1);f3=set(hist[-2])|adj(set(hist[-2]));f4=set().union(*(set(r) for r in hist[-5:-2]));outside=set(range(1,44))-(f1|f2|f3|f4)
    return f1,f2,f3,f4,outside

def shape_meta(hist):
    hall=Counter(band(r) for r in hist);c20=Counter(band(r) for r in hist[-20:]);c50=Counter(band(r) for r in hist[-50:]);out={}
    for b0 in range(7):
      for b1 in range(7-b0):
       for b2 in range(7-b0-b1):
        for b3 in range(7-b0-b1-b2):
         b4=6-b0-b1-b2-b3;sh=(b0,b1,b2,b3,b4);f=hall[sh]/len(hist)
         out[sh]={'freq':f,'layer':layer(f),'c20':c20[sh],'c50':c50[sh],'temporal_ok':c20[sh]==0 and c50[sh]<=1}
    return out

def cluster_audit(row,sets):
    R=set(row);details={};viol=0
    for name,S in sets.items():
        vals=sorted(R&S); cls=sorted({local_cluster(x) for x in vals})
        ok=(len(vals)<2 or len(cls)>=2)
        if not ok:viol+=1
        details[name]={'numbers':vals,'clusters':[c+1 for c in cls],'spread_ok':ok}
    return viol,details

def hyp(K):
    den=math.comb(43,6);a=np.zeros(7,float)
    for k in range(7):
        if k<=K and 6-k<=43-K:a[k]=math.comb(K,k)*math.comb(43-K,6-k)/den
    return a

def lex_index(row):
    rank=0;prev=0;k=6;n=43
    for i,x in enumerate(row):
        for v in range(prev+1,x):rank+=math.comb(n-v,k-i-1)
        prev=x
    return rank

def committee_scores(draws,bonus,selected):
    T=len(draws)
    C=np.fromiter((x for c in itertools.combinations(range(1,44),6) for x in c),dtype=np.int16,count=NALL*6).reshape(NALL,6)
    sums=C.sum(1);sb=np.clip((sums-21)//5,0,50).astype(np.int16);odd=(C%2).sum(1).astype(np.int8);con=(np.diff(C,axis=1)==1).sum(1).astype(np.int8)
    b0=(C<=9).sum(1);b1=((C>=10)&(C<=19)).sum(1);b2=((C>=20)&(C<=29)).sum(1);b3=((C>=30)&(C<=39)).sum(1);b4=(C>=40).sum(1)
    bcode=(b0*2401+b1*343+b2*49+b3*7+b4).astype(np.int32)
    gs=np.std(np.diff(C,axis=1),axis=1);qcuts=np.quantile(gs,[.2,.4,.6,.8]);gap=np.digitize(gs,qcuts).astype(np.int8)
    sizes={'sum':51,'odd':7,'band':16807,'consec':6,'gap':5,'prev':7,'prev2':7,'pbonus':2,'hot':7}
    pri={'sum':np.bincount(sb,minlength=51).astype(float),'odd':np.bincount(odd,minlength=7).astype(float),'band':np.bincount(bcode,minlength=16807).astype(float),'consec':np.bincount(con,minlength=6).astype(float),'gap':np.bincount(gap,minlength=5).astype(float)}
    for k in pri:pri[k]/=pri[k].sum()
    pri['prev']=hyp(6);pri['prev2']=hyp(6);pri['hot']=hyp(15);pri['pbonus']=np.array([37/43,6/43],float)
    npref=np.zeros((T+1,44),np.int32);ppref=np.zeros((T+1,44,44),np.int16)
    for u,row in enumerate(draws):
        npref[u+1]=npref[u];npref[u+1,row]+=1;ppref[u+1]=ppref[u]
        for a,b in itertools.combinations(map(int,row),2):ppref[u+1,a,b]+=1;ppref[u+1,b,a]+=1
    actual=[]
    for u,row in enumerate(draws):
        cur=set(map(int,row));pr=set(map(int,draws[u-1])) if u else set();pr2=set(map(int,draws[u-2])) if u>=2 else set();c300=npref[u]-npref[max(0,u-300)];hot=set((np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]) if u else set()
        actual.append({'sum':int(np.clip((int(row.sum())-21)//5,0,50)),'odd':int((row%2).sum()),'band':band_code_tuple(band(row)),'consec':int((np.diff(row)==1).sum()),'gap':int(np.digitize(np.std(np.diff(row)),qcuts)),'prev':len(cur&pr),'prev2':len(cur&pr2),'pbonus':int(u>=1 and int(bonus[u-1]) in cur),'hot':sum(int(x) in hot for x in row)})
    lo=max(0,T-500);W={}
    for f in sizes:
        wins=np.bincount([actual[u][f] for u in range(lo,T)],minlength=sizes[f]).astype(float);p=pri[f];q=(wins+ALPHA*p)/(T-lo+ALPHA);W[f]=np.clip(np.log(np.maximum(q,1e-15)/np.maximum(p,1e-15)),-CLIP,CLIP)
    pr=draws[-1];pr2=draws[-2];c300=npref[T]-npref[max(0,T-300)];hot=(np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]
    pf=np.zeros(44,np.int8);pf[pr]=1;p2f=np.zeros(44,np.int8);p2f[pr2]=1;hf=np.zeros(44,np.int8);hf[hot]=1
    po=pf[C].sum(1);p2=p2f[C].sum(1);hh=hf[C].sum(1);pb=(C==int(bonus[-1])).any(1).astype(np.int8)
    stat=(W['sum'][sb]+W['odd'][odd]+W['band'][bcode]+W['consec'][con]+W['gap'][gap]+W['prev'][po]+W['prev2'][p2]+W['pbonus'][pb]+W['hot'][hh]).astype(np.float32)
    pairc=ppref[T]-ppref[max(0,T-300)];c5=np.empty(NALL,np.float32);c4=np.empty(NALL,np.float32);CH=350000
    for a in range(0,NALL,CH):
        z=min(NALL,a+CH);X=C[a:z];total=np.zeros(z-a,np.float32);inc=np.zeros((z-a,6),np.float32)
        for i,j in itertools.combinations(range(6),2):
            v=pairc[X[:,i],X[:,j]].astype(np.float32);total+=v;inc[:,i]+=v;inc[:,j]+=v
        c5[a:z]=np.max(total[:,None]-inc,axis=1);v4=np.full(z-a,-1e9,np.float32)
        for i,j in itertools.combinations(range(6),2):v4=np.maximum(v4,total-inc[:,i]-inc[:,j]+pairc[X[:,i],X[:,j]])
        c4[a:z]=v4
    zs=(stat-stat.mean())/(stat.std()+1e-9);z5=(c5-c5.mean())/(c5.std()+1e-9);z4=(c4-c4.mean())/(c4.std()+1e-9);comm=(zs+.20*z5+.15*z4).astype(np.float32)
    sc=np.sort(comm)
    out=[]
    for row in selected:
        i=lex_index(row);v=float(comm[i]);rank=int(NALL-np.searchsorted(sc,v,side='right')+1);pct=100.0*(rank-1)/(NALL-1)
        out.append({'committee':v,'committee_rank':rank,'committee_percentile_from_top':pct,'stat':float(stat[i]),'core5':float(c5[i]),'core4':float(c4[i])})
    return out

def main():
    rows=load_rows();assert rows[-1][0]==2136,rows[-1][0]
    nums=[np.asarray(r[1],np.int16) for r in rows];bonus=np.asarray([r[2] for r in rows],np.int16)
    hist=[tuple(map(int,x)) for x in nums];prev=hist[-1];prevbo=int(bonus[-1]);f1,f2,f3,f4,outside=flow_sets(hist);sets={'prev_same':f1,'prev_pm1':f2,'prev2_same_pm1':f3,'draws3to5_same':f4};sm=shape_meta(hist)
    candidates=[]
    for row in itertools.combinations(range(1,44),6):
        if 23 in row or 24 in row or has_three_consecutive(row):continue
        R=set(row);c1=len(R&f1);c2=len(R&f2);c3=len(R&f3);c4=len(R&f4);co=len(R&outside)
        if not (0<=c1<=2 and 1<=c2<=2 and 1<=c3<=4 and 1<=c4<=2 and co==1):continue
        low=sum(x<=31 for x in row);high=6-low
        if low<2 or high<1:continue
        sh=band(row);m=sm[sh]
        if not m['temporal_ok']:continue
        viol,cd=cluster_audit(row,sets)
        candidates.append({'row':row,'state':sum_state(sum(row)),'layer':m['layer'],'shape':sh,'c20':m['c20'],'c50':m['c50'],'c1':c1,'c2':c2,'c3':c3,'c4':c4,'outside':next(iter(R&outside)),'bo':int(prevbo in R),'low':low,'high':high,'cluster_viol':viol,'cluster_detail':cd})
    state_plan=['S2','S2','S2','S1','S1','S1','S3','S3','S4','S4']
    layer_plan=['A','B','C','A','B','C','A','B','D','B']
    c1_plan=[0,0,1,0,1,2,0,1,1,2];c2_plan=[1,1,1,1,1,2,1,1,1,2];bo_plan=[0,0,0,0,0,0,0,0,0,1];cluster_relax=[0,0,0,0,0,1,0,0,1,1]
    c3_t=[2,3,2,2,3,2,2,3,2,3];c4_t=[2,1,2,2,1,2,1,2,2,1]
    selected=[];numuse=Counter();pairuse=Counter();triuse=Counter();prevuse=Counter()
    def key(x,pos,relax_layer=False,relax_flow=False):
        row=x['row'];pairs=list(itertools.combinations(row,2));tris=list(itertools.combinations(row,3));target={'S1':101,'S2':119,'S3':139,'S4':157}[x['state']]
        return (x['cluster_viol'],0 if (relax_layer or x['layer']==layer_plan[pos]) else 1,0 if (relax_flow or x['c1']==c1_plan[pos]) else 1,0 if (relax_flow or x['c2']==c2_plan[pos]) else 1,abs(x['c3']-c3_t[pos]),abs(x['c4']-c4_t[pos]),sum(triuse[t] for t in tris),sum(pairuse[p] for p in pairs),max((numuse[n] for n in row),default=0),sum(numuse[n] for n in row),abs(sum(row)-target),row)
    for pos,st in enumerate(state_plan):
        best=None;bestk=None
        for relax in range(3):
            for x in candidates:
                if x['state']!=st or x['bo']!=bo_plan[pos]:continue
                if any(x['row']==y['row'] for y in selected):continue
                if x['cluster_viol']>cluster_relax[pos]:continue
                if relax==0 and x['layer']!=layer_plan[pos]:continue
                if relax<2 and (x['c1']!=c1_plan[pos] or x['c2']!=c2_plan[pos]):continue
                prevnums=[n for n in x['row'] if n in f1]
                if any(prevuse[n]>=2 for n in prevnums):continue
                k=key(x,pos,relax_layer=relax>0,relax_flow=relax>1)
                if bestk is None or k<bestk:bestk=k;best=x
            if best is not None:break
        assert best is not None,(pos,st)
        selected.append(best)
        for n in best['row']:
            numuse[n]+=1
            if n in f1:prevuse[n]+=1
        for p in itertools.combinations(best['row'],2):pairuse[p]+=1
        for t in itertools.combinations(best['row'],3):triuse[t]+=1
    scores=committee_scores(np.asarray(nums,np.int16),bonus,[x['row'] for x in selected])
    out={'draw':2137,'previous':{'numbers':list(prev),'bonus':prevbo,'sum':sum(prev),'state':sum_state(sum(prev))},'local_clusters':[list(x) for x in LOCAL],'flow_sets':{'prev_same':sorted(f1),'prev_pm1':sorted(f2),'prev2_same_pm1':sorted(f3),'draws3to5_same':sorted(f4),'outside_four':sorted(outside)},'portfolio':[]}
    for i,(x,s) in enumerate(zip(selected,scores),1):
        checks={'prev_same_ok':x['c1']<=1,'prev_same_rescue':x['c1']==2,'prev_pm1_ok':x['c2']==1,'prev_pm1_rescue':x['c2']==2,'prev2_ok':1<=x['c3']<=4,'draws3to5_ok':1<=x['c4']<=2,'outside_exact1':True,'band_temporal_ok':x['c20']==0 and x['c50']<=1,'zone_ok':x['low']>=2 and x['high']>=1,'exclude_23_24':True,'no_three_consecutive':True,'previous_bonus_main_ok':not bool(x['bo']),'previous_bonus_rescue':bool(x['bo']),'cluster_spread_ok':x['cluster_viol']==0,'cluster_rescue':x['cluster_viol']>0}
        out['portfolio'].append({'no':i,'numbers':list(x['row']),'sum':sum(x['row']),'sum_state':x['state'],'layer':x['layer'],'shape':list(x['shape']),'prev_same':x['c1'],'prev_pm1':x['c2'],'prev2_same_pm1':x['c3'],'draws3to5_same':x['c4'],'outside_number':x['outside'],'previous_bonus_included':bool(x['bo']),'cluster_violations':x['cluster_viol'],'cluster_detail':x['cluster_detail'],'checks':checks,**s})
    out['audit']={'state_counts':dict(Counter(x['state'] for x in selected)),'layer_counts':dict(Counter(x['layer'] for x in selected)),'cluster_strict_lines':sum(x['cluster_viol']==0 for x in selected),'previous_bonus_lines':sum(x['bo'] for x in selected),'union_size':len(set().union(*(set(x['row']) for x in selected))),'repeated_pairs':sum(v>1 for v in pairuse.values()),'repeated_triples':sum(v>1 for v in triuse.values())}
    p=OUT/'loto6_2137_cluster_flow_committee.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
