import json,runpy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos'];draws=m['draws'];H=len(draws);idxmap={tuple(map(int,q)):i for i,q in enumerate(combos)}
if m.get('committee_score'):
 stat,core,comm=m['committee_score'](H)
else:
 stat=m['a1_score'](H);core=m['a2_score'](H);comm=m['z'](stat)+0.15*m['z'](core)
def z(x):
 x=np.asarray(x,float);return (x-x.mean())/(x.std()+1e-9)
score=z(comm)+0.40*z(stat)
winner=(1,4,20,25,29);wi=idxmap[winner];sums=combos.sum(1)
def rank(mask):
 ids=np.where(mask)[0];order=ids[np.argsort(-score[ids],kind='stable')];pos=np.where(order==wi)[0];return None if not len(pos) else int(pos[0]+1),int(len(ids))
res={'draw':1402,'winner':list(winner),'winner_sum':sum(winner),'score':'z(Committee)+0.40*z(Stat)','original_85_105':rank((sums>=85)&(sums<=105)),'revised_sum_le_105':rank(sums<=105),'unrestricted_all_169911':rank(np.ones(len(combos),bool)),'winner_stat_rank_all':int(1+np.sum(stat>stat[wi])),'winner_committee_rank_all':int(1+np.sum(comm>comm[wi])),'note':'Ranks use only history through draw 1401 to construct scores. Winner is used only afterward to locate its position.'}
p=ROOT/'data'/'miniloto-1402-initial-depth.json';p.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(res,ensure_ascii=False,indent=2))
