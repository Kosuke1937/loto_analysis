import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'data'/'miniloto-score-traces-1200-1399.json').read_text(encoding='utf-8'))['rows']
X=[r['draw'] for r in rows]
out=ROOT/'docs'/'score_pngs'; out.mkdir(parents=True,exist_ok=True)

def plot(keys,labels,name,title,ylabel='Winner Z score'):
    plt.figure(figsize=(16,6))
    for k,l in zip(keys,labels):
        y=[r.get(k) for r in rows]
        plt.plot(X,y,label=l,linewidth=1.2)
    plt.axhline(0,linewidth=.8)
    plt.xlabel('Draw')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=3,fontsize=8)
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig(out/name,dpi=170)
    plt.close()

plot(['A1','core100','core200','core300','core500','coreAll'],['A1','4core100','4core200','4core300','4core500','4core all'],'01_a1_core_windows.png','A1 and 4core window comparison')
plot(['lambda_0.00','lambda_0.05','lambda_0.10','lambda_0.15','lambda_0.20','lambda_0.30','lambda_0.50','lambda_1.00'],['0','0.05','0.10','0.15 current','0.20','0.30','0.50','1.0'],'02_lambda.png','4core300 weight lambda')
plot(['Committee015','Count200_015','Lift300_050','Under500_030','Interact005','Interact010','Interact020'],['Current','Count200 .15','Lift300 .50','Under500 .30','Interaction .05','Interaction .10','Interaction .20'],'03_alternatives.png','Alternative committee formulations')
plot(['Committee015','A_Stat','A_Dynamic','B_Stat'],['Current','A-Stat','A-Dynamic','B-Stat'],'04_specialists.png','A/B specialist comparison')
plot(['Product025','Product050','Product100'],['beta .25','beta .50','beta 1.0'],'05_product.png','Multiplicative diagnostic',ylabel='Shifted product score')
print('done')
