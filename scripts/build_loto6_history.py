import csv, io, json, os, urllib.request
from pathlib import Path

URLS=[
 'https://loto6.thekyo.jp/data/loto6.csv',
 'https://mk-mode.com/rails/loto/LOTO6_ALL.csv',
]

def get_text():
    last=None
    for url in URLS:
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            b=urllib.request.urlopen(req,timeout=30).read()
            for enc in ('cp932','shift_jis','utf-8-sig','utf-8'):
                try:
                    t=b.decode(enc)
                    if '開催回' in t or '抽せん回' in t: return t,url
                except UnicodeDecodeError: pass
        except Exception as e: last=e
    raise RuntimeError(f'download failed: {last}')

def pick(r,*names):
    for n in names:
        if n in r and str(r[n]).strip()!='': return str(r[n]).strip()
    return ''

def asint(v):
    s=str(v).replace(',','').strip()
    return int(float(s)) if s else None

def main():
    text,src=get_text()
    rd=csv.DictReader(io.StringIO(text))
    rows=[]
    for r in rd:
        try:
            draw=asint(pick(r,'開催回','抽せん回','回号','回'))
            if not draw: continue
            date=pick(r,'日付','抽せん日','抽選日').replace('-','/')
            nums=[]
            for i in range(1,7):
                v=pick(r,f'第{i}数字',f'本数字{i}',f'数字{i}')
                nums.append(asint(v))
            if any(v is None for v in nums): continue
            bonus=asint(pick(r,'BONUS数字','ボーナス数字','Bonus','B数字'))
            cnt=asint(pick(r,'1等口数','１等口数'))
            rows.append([draw,date,*sorted(nums),bonus,cnt])
        except Exception:
            continue
    by={r[0]:r for r in rows}
    # Recent draws fixed from already-verified project data / result checks.
    recent=[
      [2124,'2026/07/30',6,20,29,36,37,41,19,0],
      [2125,'2026/08/03',3,22,25,28,30,43,39,None],
      [2126,'2026/08/06',18,20,22,35,37,42,24,1],
      [2127,'2026/08/10',7,12,15,29,30,33,22,1],
      [2128,'2026/08/13',13,20,22,27,35,38,11,None],
      [2129,'2026/08/17',2,4,6,16,25,41,40,None],
    ]
    for r in recent: by[r[0]]=r
    rows=[by[i] for i in sorted(by)]
    if not rows or rows[0][0]!=1 or rows[-1][0]!=2129:
        raise RuntimeError(f'bad range: {rows[0][0] if rows else None}..{rows[-1][0] if rows else None}')
    miss=[i for i in range(1,2130) if i not in by]
    if miss: raise RuntimeError(f'missing draws: {miss[:20]} total={len(miss)}')
    if rows[-1][2:9] != [2,4,6,16,25,41,40]:
        raise RuntimeError('latest draw validation failed')
    Path('data').mkdir(exist_ok=True)
    for p in Path('data').glob('loto6-chunk-*.js'): p.unlink()
    chunk=200
    for k,start in enumerate(range(0,len(rows),chunk),1):
        part=rows[start:start+chunk]
        js='window.LOTO6_CHUNKS=window.LOTO6_CHUNKS||[];window.LOTO6_CHUNKS.push('+json.dumps(part,ensure_ascii=False,separators=(',',':'))+');\n'
        Path(f'data/loto6-chunk-{k}.js').write_text(js,encoding='utf-8')
    meta={'count':len(rows),'first':rows[0][0],'last':rows[-1][0],'latest':rows[-1],'source':src}
    Path('data/loto6-meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(meta)

if __name__=='__main__': main()
