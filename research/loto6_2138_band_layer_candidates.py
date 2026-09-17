#!/usr/bin/env python3
import json
from pathlib import Path
from loto6_2137_cluster_flow_committee import load_rows, shape_meta
ROOT=Path(__file__).resolve().parents[1]
rows=load_rows(); hist=[r[1] for r in rows]
sm=shape_meta(hist)
shapes=[(1,2,1,1,1),(2,1,1,1,1),(1,1,2,1,1),(2,2,0,2,0),(1,1,2,2,0),(1,1,0,3,1),(3,2,1,0,0),(1,3,1,1,0),(1,1,1,3,0),(2,0,2,2,0)]
out=[]
for sh in shapes:
 m=sm[sh]
 out.append({'shape':'-'.join(map(str,sh)),'freq':m['freq'],'layer':m['layer'],'c20':m['c20'],'c50':m['c50'],'temporal_ok':m['temporal_ok']})
p=ROOT/'research'/'results'/'loto6_2138_band_layer_candidates.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
