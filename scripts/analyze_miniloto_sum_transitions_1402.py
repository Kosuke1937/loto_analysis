from pathlib import Path
import re, json, math, statistics

ROOT = Path(__file__).resolve().parents[1]
rows=[]
for i in range(1,9):
    p=ROOT/'data'/f'miniloto-chunk-{i}.js'
    s=p.read_text(encoding='utf-8')
    m=re.search(r'push\((\[.*\])\);?\s*$',s,re.S)
    if not m:
        raise SystemExit(f'cannot parse {p}')
    rows.extend(json.loads(m.group(1)))
rows=sorted(rows,key=lambda r:r[0])
if rows[-1][0] != 1402:
    raise SystemExit(f'latest draw unexpected: {rows[-1][0]}')

sums=[sum(r[2:7]) for r in rows]
changes=[sums[i]-sums[i-1] for i in range(1,len(sums))]

def q(vals,p):
    x=sorted(vals)
    if not x:return None
    k=(len(x)-1)*p
    f=math.floor(k); c=math.ceil(k)
    if f==c:return x[f]
    return x[f]*(c-k)+x[c]*(k-f)

def stats(vals):
    return {
      'n':len(vals),
      'mean':statistics.fmean(vals) if vals else None,
      'sd_pop':statistics.pstdev(vals) if len(vals)>1 else None,
      'sd_sample':statistics.stdev(vals) if len(vals)>1 else None,
      'median':statistics.median(vals) if vals else None,
      'q10':q(vals,.10),'q25':q(vals,.25),'q75':q(vals,.75),'q90':q(vals,.90),
      'min':min(vals) if vals else None,'max':max(vals) if vals else None,
    }

def corr(x,y):
    mx=statistics.fmean(x); my=statistics.fmean(y)
    num=sum((a-mx)*(b-my) for a,b in zip(x,y))
    den=(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))**0.5
    return num/den if den else None

bins=[(-10**9,60,'<=60'),(61,70,'61-70'),(71,80,'71-80'),(81,90,'81-90'),(91,100,'91-100'),(101,110,'101-110'),(111,10**9,'111+')]
def bin_counts(vals):
    out={lab:0 for _,_,lab in bins}
    for v in vals:
        for lo,hi,lab in bins:
            if lo<=v<=hi:
                out[lab]+=1; break
    n=len(vals)
    return {k:{'count':v,'pct':(100*v/n if n else None)} for k,v in out.items()}

baseline={
 'sum_stats':stats(sums),
 'change_stats':stats(changes),
 'change_lag1_corr':corr(changes[:-1],changes[1:]),
 'next_sum_bins':bin_counts(sums)
}

conditions={
 'near_current': lambda a,b: 107<=a<=117 and 74<=b<=84 and -40 <= b-a <= -25,
 'high_to_mean_strict': lambda a,b: a>=108 and 71<=b<=89 and (b-a)<=-25,
 'high_to_mean_broad': lambda a,b: a>=100 and 70<=b<=90 and (b-a)<=-20,
 'central_only_near': lambda a,b: 74<=b<=84,
 'central_only_strict': lambda a,b: 71<=b<=89,
 'central_only_broad': lambda a,b: 70<=b<=90,
}

def analyze_condition(fn):
    events=[]
    for i in range(1,len(sums)-1):
        a,b,c=sums[i-1],sums[i],sums[i+1]
        if fn(a,b):
            events.append({'draw_prev2':rows[i-1][0],'sum_prev2':a,'draw_prev1':rows[i][0],'sum_prev1':b,'draw_next':rows[i+1][0],'sum_next':c,'next_change':c-b})
    ns=[e['sum_next'] for e in events]; nd=[e['next_change'] for e in events]
    rise=sum(d>0 for d in nd); fall=sum(d<0 for d in nd); same=sum(d==0 for d in nd); flat10=sum(abs(d)<=10 for d in nd)
    return {
      'n':len(events), 'next_sum_stats':stats(ns), 'next_change_stats':stats(nd), 'next_sum_bins':bin_counts(ns),
      'direction':{
        'rise_count':rise,'rise_pct':100*rise/len(events) if events else None,
        'fall_count':fall,'fall_pct':100*fall/len(events) if events else None,
        'same_count':same,'same_pct':100*same/len(events) if events else None,
        'within_pm10_count':flat10,'within_pm10_pct':100*flat10/len(events) if events else None
      },
      'events':events
    }

analyses={k:analyze_condition(v) for k,v in conditions.items()}
comparisons={}
for hi,base in [('near_current','central_only_near'),('high_to_mean_strict','central_only_strict'),('high_to_mean_broad','central_only_broad')]:
    a=analyses[hi]; b=analyses[base]
    comparisons[hi]={
      'condition_n':a['n'],'central_only_n':b['n'],
      'next_mean_diff_vs_central_only':a['next_sum_stats']['mean']-b['next_sum_stats']['mean'],
      'rise_pct_diff_vs_central_only':a['direction']['rise_pct']-b['direction']['rise_pct'],
      'condition_next_mean':a['next_sum_stats']['mean'],'central_only_next_mean':b['next_sum_stats']['mean'],
      'condition_rise_pct':a['direction']['rise_pct'],'central_only_rise_pct':b['direction']['rise_pct']
    }

result={
 'latest':{'draw_prev2':1401,'sum_prev2':sums[-2],'draw_prev1':1402,'sum_prev1':sums[-1],'change':sums[-1]-sums[-2]},
 'baseline':baseline,
 'conditions':analyses,
 'comparisons':comparisons,
 'definitions':{
   'near_current':'107<=high<=117, 74<=current<=84, drop -40..-25',
   'high_to_mean_strict':'high>=108, current 71..89, drop<=-25',
   'high_to_mean_broad':'high>=100, current 70..90, drop<=-20',
   'central_only_near':'current 74..84 regardless of prior sum',
   'central_only_strict':'current 71..89 regardless of prior sum',
   'central_only_broad':'current 70..90 regardless of prior sum'
 }
}

out=ROOT/'data'/'miniloto-sum-transition-analysis-1402.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({
 'latest':result['latest'],
 'sum_stats':baseline['sum_stats'],
 'change_stats':baseline['change_stats'],
 'change_lag1_corr':baseline['change_lag1_corr'],
 'conditions':{k:{'n':v['n'],'next_sum_stats':v['next_sum_stats'],'direction':v['direction'],'next_sum_bins':v['next_sum_bins']} for k,v in analyses.items()},
 'comparisons':comparisons
},ensure_ascii=False,indent=2))
