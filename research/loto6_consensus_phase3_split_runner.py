#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split runner for Phase3 cascade to stay under GitHub Actions time limit."""
import importlib.util, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ph3', ROOT/'research'/'loto6_consensus_phase3_cascade.py')
ph3=importlib.util.module_from_spec(spec); spec.loader.exec_module(ph3)
start=int(os.environ['PH3_START']); end=int(os.environ['PH3_END'])
ph3.p.START_DRAW=start; ph3.p.END_DRAW=end
# make split labels internally consistent
ph3.DEV_END=end if end <= 1877 else 1877
ph3.main()
# rename output for artifact clarity
src=ROOT/'research'/'results'/'loto6_consensus_phase3_cascade_summary.json'
dst=ROOT/'research'/'results'/f'loto6_consensus_phase3_cascade_{start}_{end}.json'
if src.exists(): src.replace(dst)
print(dst)
