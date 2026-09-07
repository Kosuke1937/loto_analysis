#!/usr/bin/env python3
from pathlib import Path
import json, urllib.request
ROOT=Path(__file__).resolve().parents[1]

def local_rows():
    out=[]
    for k in range(1,13):
        s=(ROOT/'data'/f'loto6-chunk-{k}.js').read_text(encoding='utf-8')
        payload=s.split('push(',1)[1].rsplit(');',1)[0]
        out.extend(json.loads(payload))
    return [r for r in out if int(r[0])<=2134]

rows=local_rows()
lines=['draw,date,n1,n2,n3,n4,n5,n6,bonus']
for r in rows:
    lines.append(','.join(map(str,[r[0],r[1],*r[2:8],r[8]])))
raw=('\n'.join(lines)+'\n').encode('cp932')
class _Resp:
    def read(self): return raw
urllib.request.urlopen=lambda *args,**kwargs:_Resp()

src=(ROOT/'research'/'loto6_2134_winner_postmortem.py').read_text(encoding='utf-8')
src=src.replace('WIN=(5,9,10,19,26,35)','WIN=(7,13,32,34,37,39)')
src=src.replace('r[0]<=2133','r[0]<=2134')
src=src.replace("'draw':2134","'draw':2135")
src=src.replace('loto6_2134_winner_postmortem.json','loto6_2135_winner_postmortem.json')
exec(compile(src,'loto6_2135_winner_postmortem_runtime.py','exec'),{'__name__':'__main__','__file__':str(ROOT/'research'/'loto6_2135_winner_postmortem_runtime.py')})
