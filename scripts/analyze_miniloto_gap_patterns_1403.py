import json,re,itertools,collections,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8')
    m=re.search(r'push\((\[.*\])\);?$',txt,re.S)
    if not m: raise SystemExit(f'cannot parse {p}')
    DATA.extend(json.loads(m.group(1)))
DATA.sort(key=lambda r:r[0])

def gv(a):
    a=sorted(a)
    return tuple(a[i+1]-a[i] for i in range(4))
def recs(rows):
    return [gv(r[2:7]) for r in rows]
def summarize(rows):
    V=recs(rows); n=len(V)
    flat=[g for v in V for g in v]
    pos=[]
    for j in range(4):
        c=collections.Counter(v[j] for v in V)
        pos.append({'position':j+1,'mean':sum(v[j] for v in V)/n,'median':statistics.median(v[j] for v in V),'top':c.most_common(8)})
    allc=collections.Counter(flat)
    vec=collections.Counter(V)
    mult=collections.Counter(tuple(sorted(v)) for v in V)
    ones=collections.Counter(sum(g==1 for g in v) for v in V)
    le3=collections.Counter(sum(g<=3 for g in v) for v in V)
    ge8=collections.Counter(sum(g>=8 for g in v) for v in V)
    rep=sum(len(set(v))<4 for v in V)
    allsmall=sum(max(v)<=5 for v in V)
    spans=[sum(v) for v in V]
    return {
      'n':n,
      'gap_frequency':{str(k):{'count':allc[k],'pct':100*allc[k]/len(flat)} for k in sorted(allc)},
      'position_stats':pos,
      'consecutive_pair_count':dict(sorted(ones.items())),
      'count_gap_le3':dict(sorted(le3.items())),
      'count_gap_ge8':dict(sorted(ge8.items())),
      'repeated_gap_any':{'count':rep,'pct':100*rep/n},
      'all_gaps_le5':{'count':allsmall,'pct':100*allsmall/n},
      'span':{'mean':sum(spans)/n,'median':statistics.median(spans),'q25':statistics.quantiles(spans,n=4,method='inclusive')[0],'q75':statistics.quantiles(spans,n=4,method='inclusive')[2], 'min':min(spans),'max':max(spans)},
      'top_exact_vectors':[{'vector':list(v),'count':c,'pct':100*c/n} for v,c in vec.most_common(20)],
      'top_unordered_gap_sets':[{'gaps_sorted':list(v),'count':c,'pct':100*c/n} for v,c in mult.most_common(20)]
    }
# combinatorial baseline across all 31C5
ALL=[[0,'',*c,0,None] for c in itertools.combinations(range(1,32),5)]
base=summarize(ALL)
periods={'all':DATA,'500':DATA[-500:],'100':DATA[-100:],'50':DATA[-50:],'20':DATA[-20:]}
out={'latest_draw':DATA[-1][0],'latest_numbers':DATA[-1][2:7],'latest_vector':list(gv(DATA[-1][2:7])),'periods':{k:summarize(v) for k,v in periods.items()},'baseline_all_combinations':base}
# current vector frequency details
cv=gv(DATA[-1][2:7])
for k,rows in periods.items():
    V=recs(rows);out.setdefault('latest_vector_frequency',{})[k]={'count':sum(v==cv for v in V),'pct':100*sum(v==cv for v in V)/len(V)}
baseV=recs(ALL);out['latest_vector_frequency']['baseline']={'count':sum(v==cv for v in baseV),'pct':100*sum(v==cv for v in baseV)/len(baseV)}
# lift per gap value recent100 vs baseline
b={int(k):v['pct'] for k,v in base['gap_frequency'].items()};r={int(k):v['pct'] for k,v in out['periods']['100']['gap_frequency'].items()}
out['recent100_gap_lift_vs_baseline']=[{'gap':g,'recent100_pct':r.get(g,0),'baseline_pct':b.get(g,0),'lift':(r.get(g,0)/b[g] if b.get(g,0)>0 else None)} for g in sorted(b)]
p=ROOT/'data'/'miniloto-gap-patterns-1403.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'latest':{'draw':out['latest_draw'],'numbers':out['latest_numbers'],'vector':out['latest_vector'],'vector_freq':out['latest_vector_frequency']},'all':out['periods']['all'],'recent100':out['periods']['100'],'recent50':out['periods']['50'],'recent20':out['periods']['20'],'baseline':base,'lift100':out['recent100_gap_lift_vs_baseline']},ensure_ascii=False,indent=2))
