import json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
g=runpy.run_path(str(ROOT/'scripts'/'analyze_miniloto_1402_rebuild_generations.py'))
evaluate=g['evaluate']; winner=g['winner']; base_t=g['base_t']; union=g['union']
variants=[]
for ov in (3,5):
    for sw in (0.15,0.25,0.35,0.45,0.60):
        for ns in (-3.0,-2.0,-1.5,-1.0,-0.5,0.0):
            variants.append(evaluate(f'CORE_ov{ov}_stat{sw:.2f}_nov{ns:.1f}',0,105,ov,sw,ns,max_rounds=200))
valid=[v for v in variants if v['winner_generation'] is not None]
best=min(valid,key=lambda v:(v['winner_generation'],v['winner_position_in_generation'],v['winner_raw_rank'])) if valid else None
summary={
 'draw':1402,'winner':list(winner),'initial10':[list(q) for q in base_t],
 'interpretation':'negative novelty_scale rewards retention of pairs/triples already present in the initial10; this is post-hoc oracle sensitivity, not forward-valid tuning.',
 'oracle_best':best,
 'top10_by_raw_rank':sorted([v for v in variants if v['winner_eligible']],key=lambda v:v['winner_raw_rank'])[:10],
 'variants_within_200_generations':sorted(valid,key=lambda v:(v['winner_generation'],v['winner_position_in_generation'],v['winner_raw_rank']))[:20]
}
(ROOT/'data'/'miniloto-1402-core-preserve-oracle-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
