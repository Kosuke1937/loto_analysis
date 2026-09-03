#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=(ROOT/'research'/'loto6_committee_2134_exact.py').read_text(encoding='utf-8')
src=src.replace("assert rows[-1][0]>=2133","assert rows[-1][0]>=2134")
src=src.replace("if r[0]<=2133","if r[0]<=2134")
src=src.replace("'target_draw':2134","'target_draw':2135")
src=src.replace("loto6_committee_2134_exact.json","loto6_committee_2135_exact.json")
exec(compile(src,'loto6_committee_2135_exact_runtime.py','exec'),{'__name__':'__main__','__file__':str(ROOT/'research'/'loto6_committee_2135_exact_runtime.py')})
