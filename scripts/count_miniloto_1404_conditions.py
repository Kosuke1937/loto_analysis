import json, re, itertools, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Load all chunk arrays exactly as app data
DATA=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8')
    m=re.search(r'push\((\[.*\])\);?$',txt,re.S)
    if not m: raise SystemExit(f'cannot parse {p}')
    DATA.extend(json.loads(m.group(1)))
DATA.sort(key=lambda r:r[0])

def band(a):
    c=[0,0,0,0]
    for n in a:c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return '-'.join(map(str,c))
def layer(freq):
    return 'A' if freq>=5 else 'B' if freq>=2 else 'C' if freq>=.5 else 'D'
def cons(a):return any(a[i]==a[i-1]+1 for i in range(1,len(a)))
latest=DATA[-1]; prev=tuple(latest[2:7]); prev2=tuple(DATA[-2][2:7]); latest_sum=sum(prev)
shapes=collections.Counter(band(r[2:7]) for r in DATA)
recent20=collections.Counter(band(r[2:7]) for r in DATA[-20:])
shape_meta={s:{'freq_pct':100*c/len(DATA),'layer':layer(100*c/len(DATA)),'recent20':recent20[s]} for s,c in shapes.items()}
eligible_shapes={s for s,m in shape_meta.items() if m['layer'] in ('A','B') and m['recent20']<=1}
prev_set=set(prev)
prev_near={n for x in prev for n in (x-1,x+1) if 1<=n<=31}
prev2_near={n for x in prev2 for n in (x-1,x,x+1) if 1<=n<=31}

def checks(c):
    c=tuple(c); ss=sum(c); sh=band(c)
    return {
      'sum': latest_sum < ss < 120,
      'shape': sh in eligible_shapes,
      'consecutive': cons(c),
      'prev_overlap': 1 <= len(set(c)&prev_set) <= 2,
      'prev_pm1': 1 <= sum(n in prev_near for n in c) <= 3,
      'prev2_ge1': sum(n in prev2_near for n in c) >= 1,
      'prev2_1to3': 1 <= sum(n in prev2_near for n in c) <= 3,
      'prev2_2to3': 2 <= sum(n in prev2_near for n in c) <= 3,
    }
allc=list(itertools.combinations(range(1,32),5))
# sequential funnel using ge1 final interpretation
order=['sum','shape','consecutive','prev_overlap','prev_pm1','prev2_ge1']
rem=allc; funnel=[]
for key in order:
    rem=[c for c in rem if checks(c)[key]]
    funnel.append({'after':key,'count':len(rem)})

def count(mode):
    keys=['sum','shape','consecutive','prev_overlap','prev_pm1',mode]
    vals=[c for c in allc if all(checks(c)[k] for k in keys)]
    byshape=collections.Counter(band(c) for c in vals)
    return {'count':len(vals),'by_shape':dict(sorted(byshape.items()))}
out={
 'latest_draw':latest[0], 'latest_numbers':list(prev), 'latest_sum':latest_sum,
 'two_back_draw':DATA[-2][0], 'two_back_numbers':list(prev2),
 'conditions':[
  'previous-main overlap 1-2', 'at least one consecutive pair',
  'band layer A/B and its band shape occurred 0 or 1 times in latest 20 draws',
  f'sum > {latest_sum} and < 120', 'count of numbers exactly +/-1 from previous mains is 1-3',
  'two-back same or +/-1: reported as >=1, 1-3, and 2-3 variants'],
 'eligible_band_shapes':[{"shape":s,**shape_meta[s]} for s in sorted(eligible_shapes)],
 'counts':{'prev2_ge1':count('prev2_ge1'),'prev2_1to3':count('prev2_1to3'),'prev2_2to3':count('prev2_2to3')},
 'funnel_prev2_ge1':funnel
}
p=ROOT/'data'/'miniloto-1404-condition-count.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
