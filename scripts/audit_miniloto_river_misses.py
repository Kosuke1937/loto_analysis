import json,re,math
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]

def load_data():
    rows=[]
    files=sorted((ROOT/'data').glob('miniloto-chunk-*.js'), key=lambda p:int(re.search(r'(\d+)',p.stem).group(1)))
    for p in files:
        s=p.read_text(encoding='utf-8')
        m=re.search(r'\.push\((\[.*\])\);?\s*$',s,re.S)
        if not m: raise RuntimeError(f'cannot parse {p}')
        rows.extend(json.loads(m.group(1)))
    rows.sort(key=lambda r:r[0])
    return rows

D=load_data()
nums=lambda i:D[i][2:7]
bonus=lambda i:D[i][7]
ALL=set(range(1,32))

def dilate(values,r=1):
    out=set()
    for x in values:
        for d in range(-r,r+1):
            if 1<=x+d<=31: out.add(x+d)
    return out

def base_set(i):
    return dilate(nums(i-1),1)|dilate(nums(i-2),1)

def prior_freq(i,w):
    c=Counter()
    for j in range(max(0,i-w),i): c.update(nums(j))
    return c

def gap_before(i,n):
    for g,j in enumerate(range(i-1,-1,-1),1):
        if n in nums(j): return g
    return i+1

def top_by_freq(i,w,k,reverse=True):
    c=prior_freq(i,w)
    arr=list(range(1,32))
    if reverse: arr.sort(key=lambda n:(-c[n],n))
    else: arr.sort(key=lambda n:(c[n],n))
    return set(arr[:k])

def overdue(i,k):
    arr=list(range(1,32));arr.sort(key=lambda n:(-gap_before(i,n),n));return set(arr[:k])

def rule_sets(i):
    b=base_set(i);outside=ALL-b
    prev12=nums(i-1)+nums(i-2)
    md={n:min(abs(n-x) for x in prev12) for n in outside}
    s3=dilate(nums(i-3),1) if i>=3 else set(); s4=dilate(nums(i-4),1) if i>=4 else set(); s5=dilate(nums(i-5),1) if i>=5 else set()
    e3=set(nums(i-3)) if i>=3 else set();e4=set(nums(i-4)) if i>=4 else set();e5=set(nums(i-5)) if i>=5 else set()
    bn=set()
    for j in (i-1,i-2):
        if j>=0 and bonus(j) is not None: bn |= dilate([bonus(j)],1)
    ld={n for n in outside if any(n%10==x%10 for x in prev12)}
    r={
      'distance2':{n for n in outside if md[n]==2},
      'distance3':{n for n in outside if md[n]==3},
      'draw3_exact':e3,
      'draw3_near1':s3,
      'draw4_near1':s4,
      'draw5_near1':s5,
      'draw3to5_exact':e3|e4|e5,
      'draw3to5_near1':s3|s4|s5,
      'bonus12_near1':bn,
      'same_last_digit_prev12':ld,
      'hot5_20':top_by_freq(i,20,5,True),
      'cold5_20':top_by_freq(i,20,5,False),
      'hot5_50':top_by_freq(i,50,5,True),
      'cold5_50':top_by_freq(i,50,5,False),
      'overdue3':overdue(i,3),
      'overdue5':overdue(i,5),
    }
    r['distance2_and_draw3to5_near1']=r['distance2']&r['draw3to5_near1']
    r['distance2_or_draw3to5_near1']=r['distance2']|r['draw3to5_near1']
    r['draw3to5_near1_or_bonus12']=r['draw3to5_near1']|r['bonus12_near1']
    r['distance2_and_bonus12']=r['distance2']&r['bonus12_near1']
    return {k:(v&outside) for k,v in r.items()}

def hg_var(N,K,n):
    if N<=1:return 0.0
    p=K/N
    return n*p*(1-p)*(N-n)/(N-1)

def base_summary(indices):
    obs=exp=var=0.0; sizes=[]; dist=Counter()
    for i in indices:
        b=base_set(i);w=set(nums(i));h=len(w&b);k=len(b)
        obs+=h;exp+=5*k/31;var+=hg_var(31,k,5);sizes.append(k);dist[h]+=1
    n=len(indices);z=(obs-exp)/math.sqrt(var) if var>0 else None
    return {'draws':n,'winner_numbers':5*n,'covered':int(obs),'coverage_rate':obs/(5*n) if n else None,'mean_hit_per_draw':obs/n if n else None,'mean_candidate_count':sum(sizes)/n if n else None,'expected_hits':exp,'expected_hit_per_draw':exp/n if n else None,'lift_vs_size_expectation':obs/exp if exp else None,'z_vs_hypergeom':z,'draw_hit_distribution':{str(k):dist[k] for k in range(6)},'draw_3plus_rate':sum(v for k,v in dist.items() if k>=3)/n if n else None,'draw_4plus_rate':sum(v for k,v in dist.items() if k>=4)/n if n else None,'draw_5of5_rate':dist[5]/n if n else None}

def rule_summary(indices):
    names=list(rule_sets(max(indices[0],5)).keys()) if indices else []
    agg={k:{'obs':0,'exp':0.0,'var':0.0,'add':0,'outside':0,'miss':0,'draws_with_rescue':0} for k in names}
    for i in indices:
        b=base_set(i);o=ALL-b;w=set(nums(i));m=w&o;rs=rule_sets(i)
        for name,A in rs.items():
            A=A&o;x=len(m&A);N=len(o);K=len(A);nn=len(m)
            a=agg[name];a['obs']+=x;a['add']+=K;a['outside']+=N;a['miss']+=nn
            if x:a['draws_with_rescue']+=1
            if N:
                a['exp']+=nn*K/N;a['var']+=hg_var(N,K,nn)
    out={}
    n=len(indices)
    for name,a in agg.items():
        z=(a['obs']-a['exp'])/math.sqrt(a['var']) if a['var']>0 else None
        out[name]={'misses_rescued':a['obs'],'total_misses':a['miss'],'miss_recall':a['obs']/a['miss'] if a['miss'] else None,'avg_added_candidates':a['add']/n if n else None,'candidate_exposures':a['add'],'expected_rescues_conditional':a['exp'],'lift_vs_random_outside':a['obs']/a['exp'] if a['exp'] else None,'z_vs_conditional_random':z,'draws_with_at_least_one_rescue':a['draws_with_rescue']}
    return out

def miss_features(indices):
    rec=[]
    for i in indices:
        b=base_set(i);prev12=nums(i-1)+nums(i-2);f20=prior_freq(i,20);f50=prior_freq(i,50)
        for n in nums(i):
            if n in b:continue
            rec.append({'draw':D[i][0],'number':n,'min_dist_prev12':min(abs(n-x) for x in prev12),'gap':gap_before(i,n),'freq20':f20[n],'freq50':f50[n],'band':'1-9' if n<=9 else '10-19' if n<=19 else '20-29' if n<=29 else '30-31','draw3_near':n in dilate(nums(i-3),1),'draw4_near':n in dilate(nums(i-4),1),'draw5_near':n in dilate(nums(i-5),1),'bonus12_near':n in (dilate([bonus(i-1)],1)|dilate([bonus(i-2)],1))})
    def count(key):return dict(sorted(Counter(r[key] for r in rec).items(),key=lambda x:str(x[0])))
    gaps=Counter()
    for r in rec:
        g=r['gap'];bucket='2' if g==2 else '3' if g==3 else '4' if g==4 else '5-7' if g<=7 else '8-12' if g<=12 else '13+'
        gaps[bucket]+=1
    return {'n_missed_numbers':len(rec),'min_distance_prev12':count('min_dist_prev12'),'gap_bucket':dict(gaps),'freq20':count('freq20'),'band':count('band'),'near_draw3_rate':sum(r['draw3_near'] for r in rec)/len(rec) if rec else None,'near_draw4_rate':sum(r['draw4_near'] for r in rec)/len(rec) if rec else None,'near_draw5_rate':sum(r['draw5_near'] for r in rec)/len(rec) if rec else None,'near_bonus12_rate':sum(r['bonus12_near'] for r in rec)/len(rec) if rec else None,'records':rec}

def latest_detail(i):
    b=base_set(i);w=set(nums(i));miss=sorted(w-b);hit=sorted(w&b);rs=rule_sets(i)
    return {'draw':D[i][0],'winner':nums(i),'base_candidates':sorted(b),'base_candidate_count':len(b),'covered':hit,'missed':miss,'rules_for_each_missed':{str(n):[name for name,A in rs.items() if n in A] for n in miss}}

valid=list(range(5,len(D)))
ranges={
 'full':valid,
 'last500':valid[-500:],
 'last200':valid[-200:],
 'last100':valid[-100:],
 'last50':valid[-50:],
}
out={'definition':'River 3-button union = numbers within ±1 of any main number in either of the previous 2 draws. Miss analysis evaluates only winning numbers outside that base set. Rescue-rule lift is conditional on the outside candidate pool, controlling for how many extra numbers each rule adds.','latest_draw':D[-1][0],'ranges':{},'latest_detail':latest_detail(len(D)-1)}
for name,idx in ranges.items():
    out['ranges'][name]={'start_draw':D[idx[0]][0],'end_draw':D[idx[-1]][0],'base':base_summary(idx),'rescue_rules':rule_summary(idx),'miss_features':miss_features(idx)}
# compact ranking tables excluding huge records
for name in out['ranges']:
    rr=out['ranges'][name]['rescue_rules']
    out['ranges'][name]['top_rules_by_lift']=[{'rule':k,**v} for k,v in sorted(rr.items(),key=lambda kv:(-(kv[1]['lift_vs_random_outside'] or -999),kv[1]['avg_added_candidates']))[:10]]
    out['ranges'][name]['top_rules_by_rescues_per_added']=[{'rule':k,**v} for k,v in sorted(rr.items(),key=lambda kv:-(kv[1]['misses_rescued']/kv[1]['candidate_exposures'] if kv[1]['candidate_exposures'] else 0))[:10]]

p=ROOT/'data'/'miniloto-river-miss-audit.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
# write compact summary separately
compact={'definition':out['definition'],'latest_detail':out['latest_detail'],'ranges':{}}
for name,r in out['ranges'].items():
    compact['ranges'][name]={'start_draw':r['start_draw'],'end_draw':r['end_draw'],'base':r['base'],'miss_features':{k:v for k,v in r['miss_features'].items() if k!='records'},'top_rules_by_lift':r['top_rules_by_lift'][:8],'top_rules_by_rescues_per_added':r['top_rules_by_rescues_per_added'][:8]}
q=ROOT/'data'/'miniloto-river-miss-audit-summary.json';q.write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(compact,ensure_ascii=False,indent=2))
