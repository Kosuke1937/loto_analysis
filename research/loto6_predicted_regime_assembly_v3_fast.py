#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v3',ROOT/'research'/'loto6_predicted_regime_assembly_v3.py')
v3=importlib.util.module_from_spec(spec);spec.loader.exec_module(v3)

_cache={}

def build_labels(draws):
    key=id(draws)
    if key in _cache:return _cache[key]
    shapes=[v3.v4.band_tuple(r) for r in draws]
    sums=[v3.sum_bin(int(np.sum(r))) for r in draws]
    cnt=Counter(); labels=[]
    for t,sh in enumerate(shapes):
        denom=max(1,min(500,t))
        freq=cnt.get(sh,0)/denom
        labels.append((t,v3.v2.band_class(freq),sums[t]))
        cnt[sh]+=1
        if t>=500:
            old=shapes[t-500];cnt[old]-=1
            if cnt[old]<=0:del cnt[old]
    _cache[key]=labels
    return labels

def fast_hist_labels(draws,t):
    labels=build_labels(draws)
    return labels[max(501,t-500):t]

v3.hist_labels=fast_hist_labels

if __name__=='__main__':
    v3.main()
