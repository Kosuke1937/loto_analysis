#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from collections import Counter
from pathlib import Path
import itertools

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m',ROOT/'research'/'loto6_pool_coverage_scorematch_compare.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fast_select_coverage(ids,dist,C,layers,shapes,pool,sup,n=10):
    if len(ids)==0:return []
    order_pos=dist.argsort()[:min(4000,len(ids))]
    ord0=[int(ids[j]) for j in order_pos]
    distmap={int(ids[j]):float(dist[j]) for j in order_pos}
    sel=[];used=set();lc=Counter();sc=Counter();nc=Counter();pc=Counter();tc=Counter();poolset=set(pool)
    for strict in (True,False):
        while len(sel)<n:
            best=None;bestkey=None
            for i in ord0:
                if i in sel:continue
                if not m.admissible(i,C,layers[i],shapes[i],lc,sc,nc,pc,tc,strict):continue
                row=set(map(int,C[i]));new=(row&poolset)-used
                key=(len(new),sum(sup[x] for x in new),-distmap[i])
                if bestkey is None or key>bestkey:bestkey=key;best=i
            if best is None:break
            m.add_sel(best,C,layers[best],shapes[best],sel,lc,sc,nc,pc,tc);used.update(map(int,C[best]))
        if len(sel)>=n:break
    return sel

m.select_coverage=fast_select_coverage
if __name__=='__main__':m.main()
