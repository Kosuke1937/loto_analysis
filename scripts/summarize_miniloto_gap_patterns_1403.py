import json,re,itertools,collections,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8');m=re.search(r'push\((\[.*\])\);?$',txt,re.S)
    if m: DATA.extend(json.loads(m.group(1)))
DATA.sort(key=lambda r:r[0])
def gaps(a): a=sorted(a); return tuple(a[i+1]-a[i] for i in range(4))
def calc(vs):
    n=len(vs); flat=[x for v in vs for x in v]; c=collections.Counter(flat);sp=[sum(v) for v in vs]
    return {'n':n,'gap_pct':{str(i):round(100*c[i]/len(flat),3) for i in range(1,16)},
    'position_mean':[round(sum(v[j] for v in vs)/n,3) for j in range(4)],'position_median':[statistics.median(v[j] for v in vs) for j in range(4)],
    'span_mean':round(sum(sp)/n,3),'span_median':statistics.median(sp),'span_q25':statistics.quantiles(sp,n=4,method='inclusive')[0],'span_q75':statistics.quantiles(sp,n=4,method='inclusive')[2],
    'at_least_one_gap1_pct':round(100*sum(1 in v for v in vs)/n,3),'at_least_two_gaps_le3_pct':round(100*sum(sum(x<=3 for x in v)>=2 for v in vs)/n,3),
    'repeated_gap_pct':round(100*sum(len(set(v))<4 for v in vs)/n,3),'all_gaps_le5_pct':round(100*sum(max(v)<=5 for v in vs)/n,3),'has_gap_ge8_pct':round(100*sum(max(v)>=8 for v in vs)/n,3),
    'top_vectors':[{'v':list(v),'count':k} for v,k in collections.Counter(vs).most_common(10)],
    'top_sorted_gapsets':[{'v':list(v),'count':k} for v,k in collections.Counter(tuple(sorted(v)) for v in vs).most_common(10)]}
WIN={k:calc([gaps(r[2:7]) for r in rows]) for k,rows in {'all':DATA,'500':DATA[-500:],'100':DATA[-100:],'50':DATA[-50:],'20':DATA[-20:]}.items()}
ALL=list(itertools.combinations(range(1,32),5)); BASE=calc([gaps(c) for c in ALL])
# recreate 10226 current candidate pool
prev=tuple(DATA[-1][2:7]);prev2=tuple(DATA[-2][2:7]);latest_sum=sum(prev)
def band(a):
    z=[0,0,0,0]
    for x in a:z[0 if x<=9 else 1 if x<=19 else 2 if x<=29 else 3]+=1
    return '-'.join(map(str,z))
def cons(a):return any(a[i]==a[i-1]+1 for i in range(1,5))
shapes=collections.Counter(band(r[2:7]) for r in DATA); recent20=collections.Counter(band(r[2:7]) for r in DATA[-20:])
def layer(freq):return 'A' if freq>=5 else 'B' if freq>=2 else 'C' if freq>=.5 else 'D'
elig={s for s,c in shapes.items() if layer(100*c/len(DATA)) in ('A','B') and recent20[s]<=1}
ps=set(prev);pm={n for x in prev for n in (x-1,x+1) if 1<=n<=31};p2={n for x in prev2 for n in (x-1,x,x+1) if 1<=n<=31}
C=[]
for c in ALL:
    if not(latest_sum<sum(c)<120):continue
    if band(c) not in elig or not cons(c):continue
    if not 1<=len(set(c)&ps)<=2:continue
    if not 1<=sum(x in pm for x in c)<=3:continue
    if not 2<=sum(x in p2 for x in c)<=3:continue
    C.append(c)
CV=calc([gaps(c) for c in C])
latestv=gaps(DATA[-1][2:7])
out={'latest':{'draw':DATA[-1][0],'numbers':DATA[-1][2:7],'gap_vector':list(latestv),'exact_vector_count_all':sum(gaps(r[2:7])==latestv for r in DATA),'exact_vector_count_last100':sum(gaps(r[2:7])==latestv for r in DATA[-100:])},'winners':WIN,'uniform_baseline':BASE,'candidate_10226':CV,'candidate_count':len(C)}
(ROOT/'data'/'miniloto-gap-summary-1403.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
