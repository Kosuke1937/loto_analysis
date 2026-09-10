import json,re,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8')
    m=re.search(r'push\((\[.*\])\);?$',txt,re.S)
    if not m:
        raise SystemExit(f'cannot parse {p}')
    DATA.extend(json.loads(m.group(1)))
DATA.sort(key=lambda r:r[0])
by=collections.defaultdict(list)
for r in DATA:
    key=tuple(sorted(map(int,r[2:7])))
    by[key].append({'draw':int(r[0]),'date':r[1]})
repeats=[]
for nums,occ in by.items():
    if len(occ)>=2:
        repeats.append({'numbers':list(nums),'count':len(occ),'occurrences':occ})
repeats.sort(key=lambda x:(-x['count'],x['occurrences'][0]['draw']))
out={
  'draws':len(DATA),
  'unique_combinations':len(by),
  'repeat_combination_count':len(repeats),
  'repeated_draw_occurrences':sum(x['count'] for x in repeats),
  'extra_occurrences_beyond_first':sum(x['count']-1 for x in repeats),
  'max_repeat_count':max([x['count'] for x in repeats],default=1),
  'repeats':repeats
}
p=ROOT/'data'/'miniloto-exact-repeat-analysis.json'
p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
