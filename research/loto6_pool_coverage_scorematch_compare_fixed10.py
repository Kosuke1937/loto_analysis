#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m',ROOT/'research'/'loto6_pool_coverage_scorematch_compare.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
_orig_select_order=m.select_order

def state_from(sel,C,layers,shapes):
    lc=Counter();sc=Counter();nc=Counter();pc=Counter();tc=Counter()
    for i in sel:
        row=tuple(map(int,C[i]));lc[layers[i]]+=1;sc[shapes[i]]+=1
        for x in row:nc[x]+=1
        for x in itertools.combinations(row,2):pc[x]+=1
        for x in itertools.combinations(row,3):tc[x]+=1
    return lc,sc,nc,pc,tc

def relaxed_ok(i,C,shape,sc,nc,pc,tc,strict=True,shape_cap=True):
    row=tuple(map(int,C[i]));pa=list(itertools.combinations(row,2));tr=list(itertools.combinations(row,3))
    if shape_cap and sc[shape]>=2:return False
    if strict and (any(nc[x]>=4 for x in row) or any(pc[x]>=2 for x in pa) or any(tc[x]>=1 for x in tr)):return False
    return True

def add_relaxed(i,C,shape,sel,sc,nc,pc,tc):
    row=tuple(map(int,C[i]));sel.append(int(i));sc[shape]+=1
    for x in row:nc[x]+=1
    for x in itertools.combinations(row,2):pc[x]+=1
    for x in itertools.combinations(row,3):tc[x]+=1

def fixed_select_order(order,C,layers,shapes,n=10):
    sel=_orig_select_order(order,C,layers,shapes,n)
    if len(sel)>=n:return sel[:n]
    _,sc,nc,pc,tc=state_from(sel,C,layers,shapes)
    for strict,shape_cap in ((True,True),(False,True),(False,False)):
        for ii in order:
            i=int(ii)
            if i in sel:continue
            if not relaxed_ok(i,C,shapes[i],sc,nc,pc,tc,strict,shape_cap):continue
            add_relaxed(i,C,shapes[i],sel,sc,nc,pc,tc)
            if len(sel)>=n:return sel[:n]
    return sel[:n]

def fixed_coverage(ids,dist,C,layers,shapes,pool,sup,n=10):
    if len(ids)==0:return []
    pos=dist.argsort()[:min(4000,len(ids))];ord0=[int(ids[j]) for j in pos];distmap={int(ids[j]):float(dist[j]) for j in pos}
    sel=[];used=set();lc=Counter();sc=Counter();nc=Counter();pc=Counter();tc=Counter();poolset=set(pool)
    # target ABCD quota first
    for strict in (True,False):
        while len(sel)<n:
            best=None;bestkey=None
            for i in ord0:
                if i in sel:continue
                if not m.admissible(i,C,layers[i],shapes[i],lc,sc,nc,pc,tc,strict):continue
                row=set(map(int,C[i]));new=(row&poolset)-used;key=(len(new),sum(sup[x] for x in new),-distmap[i])
                if bestkey is None or key>bestkey:bestkey=key;best=i
            if best is None:break
            m.add_sel(best,C,layers[best],shapes[best],sel,lc,sc,nc,pc,tc);used.update(map(int,C[best]))
        if len(sel)>=n:return sel[:n]
    # if quota cannot fill 10 in the 60k approximation, relax layer quota only
    _,sc,nc,pc,tc=state_from(sel,C,layers,shapes)
    for strict,shape_cap in ((True,True),(False,True),(False,False)):
        while len(sel)<n:
            best=None;bestkey=None
            for i in ord0:
                if i in sel:continue
                if not relaxed_ok(i,C,shapes[i],sc,nc,pc,tc,strict,shape_cap):continue
                row=set(map(int,C[i]));new=(row&poolset)-used;key=(len(new),sum(sup[x] for x in new),-distmap[i])
                if bestkey is None or key>bestkey:bestkey=key;best=i
            if best is None:break
            add_relaxed(best,C,shapes[best],sel,sc,nc,pc,tc);used.update(map(int,C[best]))
        if len(sel)>=n:return sel[:n]
    return sel[:n]

m.select_order=fixed_select_order
m.select_coverage=fixed_coverage
if __name__=='__main__':m.main()
