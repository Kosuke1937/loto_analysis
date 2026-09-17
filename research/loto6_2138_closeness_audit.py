#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math
from collections import Counter
from pathlib import Path
import numpy as np

from loto6_2137_cluster_flow_committee import load_rows, flow_sets, shape_meta, band, sum_state, has_three_consecutive, cluster_audit
from loto6_2138_overlap_flow import multiplicity_sets

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)
WIN=(9,16,21,26,38,40); W=set(WIN)
PURCHASED=[
(1,12,26,33,41,43),(6,16,29,31,36,38),(3,19,27,28,37,42),(11,21,22,25,30,40),(2,8,20,27,39,43),
(8,13,15,33,34,36),(6,7,10,25,27,41),(3,4,14,18,38,40),(1,2,4,11,17,42),(3,9,12,20,30,33)]

def hist_overlap(rows):
 c=Counter(len(set(r)&W) for r in rows)
 return {str(k):c[k] for k in range(7)}

def main():
 rows=load_rows(); assert rows[-1][0]==2137
 hist=[r[1] for r in rows]
 (f1,f2,f3,f4,outside),mult=multiplicity_sets(hist)
 sets={'prev_same':f1,'prev_pm1':f2,'prev2_same_pm1':f3,'draws3to5_same':f4}
 sm=shape_meta(hist)
 original=[]; relax=[]; macro=[]; exact=[]
 for row in itertools.combinations(range(1,44),6):
  if 23 in row or 24 in row or has_three_consecutive(row): continue
  R=set(row); c1=len(R&f1);c2=len(R&f2);c3=len(R&f3);c4=len(R&f4);co=len(R&outside)
  if not (0<=c1<=2 and 1<=c2<=2 and 1<=c3<=4): continue
  if sum(x<=31 for x in row)<2 or sum(x>=32 for x in row)<1: continue
  sh=band(row); meta=sm[sh]
  if not (meta['c20']==0 and meta['c50']<=1): continue
  viol,_=cluster_audit(row,sets)
  high=sum(1 for n in row if mult[n]>=3)
  # broad original mother used before portfolio slot selection
  if 1<=c4<=2:
   original.append(row)
  # one-condition relaxed mother: allow Flow4=3
  if 1<=c4<=3:
   relax.append(row)
   # actual macro cell selected after result: A/S4/outside=2, strict clusters, high-overlap<=1
   if meta['layer']=='A' and sum_state(sum(row))=='S4' and co==2 and viol==0 and high<=1:
    macro.append(row)
   # exact realized structural cell
   if (sh==(1,1,2,1,1) and c1==0 and c2==2 and c3==2 and c4==3 and co==2 and viol==0 and high<=1):
    exact.append(row)

 purchased_hits=[len(set(r)&W) for r in PURCHASED]
 result={
  'winner':list(WIN),
  'all_combinations':math.comb(43,6),
  'original_mother':{'n':len(original),'winner_in':WIN in original,'overlap_hist':hist_overlap(original),'max_overlap':max(len(set(r)&W) for r in original)},
  'relax_flow4_to3_mother':{'n':len(relax),'winner_in':WIN in relax,'overlap_hist':hist_overlap(relax),'max_overlap':max(len(set(r)&W) for r in relax)},
  'actual_macro_cell_A_S4_out2':{'n':len(macro),'winner_in':WIN in macro,'overlap_hist':hist_overlap(macro),'max_overlap':max(len(set(r)&W) for r in macro)},
  'exact_realized_cell':{'conditions':{'shape':[1,1,2,1,1],'state':'S4','layer':'A','flow1':0,'flow2':2,'flow3':2,'flow4':3,'outside':2,'cluster_viol':0,'high_overlap_max':1},'n':len(exact),'winner_in':WIN in exact,'overlap_hist':hist_overlap(exact),'max_overlap':max(len(set(r)&W) for r in exact),'five_or_six':[list(r) for r in exact if len(set(r)&W)>=5]},
  'purchased':{'n':10,'best_overlap':max(purchased_hits),'hits':purchased_hits,'union_recall':len(set().union(*map(set,PURCHASED))&W)},
  'portfolio_slots':[
   {'slot':1,'state':'S4','layer':'C','outside':1},{'slot':2,'state':'S4','layer':'B','outside':2},{'slot':3,'state':'S4','layer':'A','outside':0},
   {'slot':4,'state':'S3','layer':'C','outside':3},{'slot':5,'state':'S3','layer':'B','outside':1},{'slot':6,'state':'S3','layer':'C','outside':1},
   {'slot':7,'state':'S2','layer':'B','outside':0},{'slot':8,'state':'S2','layer':'C','outside':1},{'slot':9,'state':'S1','layer':'D','outside':2},{'slot':10,'state':'S1','layer':'A','outside':2}
  ],
  'actual_macro_tuple':{'state':'S4','layer':'A','outside':2,'covered_by_slot':False}
 }
 p=OUT/'loto6_2138_closeness_audit.json'; p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
