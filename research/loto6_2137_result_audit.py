#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
modpath=ROOT/'research'/'loto6_2137_cluster_flow_committee.py'
spec=importlib.util.spec_from_file_location('flow2137',modpath)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

WIN=(4,8,10,25,28,33); BONUS=9
PORT=[
(1,2,12,29,36,39),(4,9,13,19,34,39),(8,10,12,16,30,43),(3,12,13,15,20,36),(1,6,13,21,26,40),
(3,5,7,11,34,41),(9,11,12,28,39,40),(2,8,20,31,39,41),(10,13,22,29,40,43),(8,9,27,33,37,42)]

def main():
    rows=m.load_rows(); assert rows[-1][0]==2136
    hist=[tuple(r[1]) for r in rows]; bonuses=np.asarray([r[2] for r in rows],np.int16); draws=np.asarray(hist,np.int16)
    f1,f2,f3,f4,outside=m.flow_sets(hist); R=set(WIN)
    sm=m.shape_meta(hist); sh=m.band(WIN); meta=sm[sh]
    viol,detail=m.cluster_audit(WIN,{'prev_same':f1,'prev_pm1':f2,'prev2_same_pm1':f3,'draws3to5_same':f4,'outside_four':outside})
    score=m.committee_scores(draws,bonuses,[WIN])[0]
    matches=[]
    for i,p in enumerate(PORT,1):
        hit=sorted(set(p)&R); matches.append({'no':i,'numbers':p,'match_count':len(hit),'matched':hit})
    union=set().union(*map(set,PORT));
    out={
      'draw':2137,'winner':WIN,'bonus':BONUS,'sum':sum(WIN),'sum_state':m.sum_state(sum(WIN)),'shape':sh,'shape_c20':meta['c20'],'shape_c50':meta['c50'],'shape_long_freq':meta['freq'],'shape_layer':meta['layer'],
      'conditions':{
        'prev_same':len(R&f1),'prev_pm1':len(R&f2),'prev2_same_pm1':len(R&f3),'draws3to5_same':len(R&f4),'outside_four':len(R&outside),
        'outside_numbers':sorted(R&outside),'zone_low31':sum(x<=31 for x in WIN),'zone_high32_43':sum(x>=32 for x in WIN),
        'contains_23_24':bool(R&{23,24}),'three_consecutive':m.has_three_consecutive(WIN),'previous_bonus42_included':42 in R,
        'cluster_violations':viol,'cluster_detail':detail},
      'committee':score,
      'portfolio_matches':matches,'portfolio_best_match':max(x['match_count'] for x in matches),'portfolio_union_size':len(union),'winner_number_recall':len(union&R),'winner_numbers_in_union':sorted(union&R),'winner_numbers_missing_union':sorted(R-union),
    }
    print(json.dumps(out,ensure_ascii=False,indent=2))
    op=ROOT/'research'/'results'/'loto6_2137_result_audit.json';op.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
