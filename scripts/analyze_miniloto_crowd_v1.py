from pathlib import Path
import json, re, math, itertools, statistics
from collections import Counter, defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, balanced_accuracy_score,
    roc_auc_score, confusion_matrix
)
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / 'data' / 'miniloto-crowd-v1.json'
OUT_CSV = ROOT / 'data' / 'miniloto-crowd-v1-test.csv'


def load_rows():
    rows=[]
    for i in range(1,9):
        p=ROOT/'data'/f'miniloto-chunk-{i}.js'
        t=p.read_text(encoding='utf-8')
        m=re.search(r'push\((\[.*\])\);?\s*$',t,re.S)
        if not m:
            raise RuntimeError(p)
        rows.extend(json.loads(m.group(1)))
    rows.sort(key=lambda r:r[0])
    return rows


def band_shape(nums):
    c=[0,0,0,0]
    for n in nums:
        c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return tuple(c)


def layer_from_freq(pct):
    if pct >= 5: return 'A'
    if pct >= 2: return 'B'
    if pct >= .5: return 'C'
    return 'D'


def count_layer(n):
    if n <= 1: return '0-1'
    if n <= 5: return '1:2-5'
    if n <= 15: return '2:6-15'
    if n <= 49: return '3:16-49'
    return '4:50+'


def feature_names():
    return [
        'sum','min','max','range','odd','low12','low19',
        'b1_1_9','b2_10_19','b3_20_29','b4_30_31','occupied_bands',
        'consecutive_pairs','gap_std','max_gap','min_gap','same_last_digit_pairs',
        'prev_overlap','prev_bonus_in','shape_prior_freq_pct',
        'layer_A','layer_B','layer_C','layer_D',
        'prev20_count_median','prev20_count_mean'
    ]


def build_dataset(rows):
    shape_seen=Counter()
    counts_seen=[]
    prev_nums=None
    prev_bonus=None
    X=[]; y=[]; metas=[]
    names=feature_names()

    for idx,r in enumerate(rows):
        draw=int(r[0]); nums=list(map(int,r[2:7])); bonus=int(r[7]); cnt=r[8] if len(r)>8 else None
        shape=band_shape(nums)
        prior_n=idx
        shape_freq_pct=(100.0*shape_seen[shape]/prior_n) if prior_n>=100 else np.nan
        layer=layer_from_freq(shape_freq_pct) if not np.isnan(shape_freq_pct) else 'NA'

        gaps=[nums[i+1]-nums[i] for i in range(4)]
        same_last=sum(1 for a,b in itertools.combinations(nums,2) if a%10==b%10)
        bands=list(shape)
        prev_overlap=sum(n in set(prev_nums or []) for n in nums)
        prev_bonus_in=int(prev_bonus in nums) if prev_bonus is not None else 0
        last20=[c for c in counts_seen[-20:] if c is not None]
        med20=float(statistics.median(last20)) if last20 else np.nan
        mean20=float(statistics.mean(last20)) if last20 else np.nan

        vals={
            'sum':sum(nums),'min':min(nums),'max':max(nums),'range':max(nums)-min(nums),
            'odd':sum(n%2 for n in nums),'low12':sum(n<=12 for n in nums),'low19':sum(n<=19 for n in nums),
            'b1_1_9':bands[0],'b2_10_19':bands[1],'b3_20_29':bands[2],'b4_30_31':bands[3],
            'occupied_bands':sum(x>0 for x in bands),'consecutive_pairs':sum(g==1 for g in gaps),
            'gap_std':float(np.std(gaps)),'max_gap':max(gaps),'min_gap':min(gaps),
            'same_last_digit_pairs':same_last,'prev_overlap':prev_overlap,'prev_bonus_in':prev_bonus_in,
            'shape_prior_freq_pct':shape_freq_pct,
            'layer_A':1 if layer=='A' else 0,'layer_B':1 if layer=='B' else 0,
            'layer_C':1 if layer=='C' else 0,'layer_D':1 if layer=='D' else 0,
            'prev20_count_median':med20,'prev20_count_mean':mean20,
        }

        # Require enough history for leakage-safe prior shape frequency and exposure proxy.
        if cnt is not None and prior_n>=100 and last20:
            X.append([vals[n] for n in names])
            y.append(int(cnt))
            metas.append({
                'draw':draw,'date':r[1],'nums':nums,'bonus':bonus,'count':int(cnt),
                'count_layer':count_layer(int(cnt)),'band':'-'.join(map(str,shape)),
                'abcd':layer,'shape_prior_freq_pct':shape_freq_pct
            })

        shape_seen[shape]+=1
        counts_seen.append(None if cnt is None else int(cnt))
        prev_nums=nums; prev_bonus=bonus

    return np.array(X,dtype=float), np.array(y,dtype=int), metas, names


def split_indices(metas):
    train=np.array([i for i,m in enumerate(metas) if m['draw']<=999])
    val=np.array([i for i,m in enumerate(metas) if 1000<=m['draw']<=1199])
    test=np.array([i for i,m in enumerate(metas) if 1200<=m['draw']<=1399])
    return train,val,test


def regression_eval(y_true, pred):
    rho,p=spearmanr(y_true,pred)
    return {
        'n':int(len(y_true)),
        'mae':float(mean_absolute_error(y_true,pred)),
        'median_abs_error':float(np.median(np.abs(y_true-pred))),
        'spearman_rho':float(rho),
        'spearman_p':float(p),
    }


def layer_summary(metas):
    d=defaultdict(list)
    for m in metas:
        d[m['count_layer']].append(m)
    out={}
    for k,v in d.items():
        out[k]={
            'n':len(v),
            'count_median':float(np.median([x['count'] for x in v])),
            'sum_median':float(np.median([sum(x['nums']) for x in v])),
            'abcd_counts':dict(Counter(x['abcd'] for x in v)),
        }
    return out


def main():
    rows=load_rows()
    X,y,metas,names=build_dataset(rows)
    train,val,test=split_indices(metas)
    assert len(test)==200, len(test)

    # Impute from train only.
    med=np.nanmedian(X[train],axis=0)
    Xi=X.copy()
    bad=np.where(np.isnan(Xi))
    Xi[bad]=med[bad[1]]

    # Shape-only columns exclude exposure proxy (last two cols).
    shape_cols=np.arange(len(names)-2)
    full_cols=np.arange(len(names))

    results={
        'model':'MiniLoto Crowd-v1',
        'target_note':'Raw first-prize winning-ticket count. Sales amount is not present in the current app dataset; prev20 count median/mean are used only as exposure proxies.',
        'splits':{'train':'draw 101-999','validation':'1000-1199','fixed_test':'1200-1399'},
        'feature_names':names,
        'layer_summary_all':layer_summary(metas),
    }

    # Regression: log1p(count), Ridge. Choose alpha on validation, then freeze for test.
    alphas=[0.1,1.0,10.0,30.0,100.0]
    reg_models={}
    for label,cols in [('shape_only',shape_cols),('shape_plus_exposure',full_cols)]:
        sc=StandardScaler().fit(Xi[train][:,cols])
        Xtr=sc.transform(Xi[train][:,cols]); Xv=sc.transform(Xi[val][:,cols]); Xt=sc.transform(Xi[test][:,cols])
        best=None
        for a in alphas:
            md=Ridge(alpha=a).fit(Xtr,np.log1p(y[train]))
            pv=np.expm1(md.predict(Xv))
            mae=mean_absolute_error(y[val],pv)
            if best is None or mae<best[0]: best=(mae,a,md)
        _,alpha,md=best
        pt=np.expm1(md.predict(Xt))
        pt=np.clip(pt,0,None)
        coefs=dict(sorted(zip([names[i] for i in cols],md.coef_.tolist()),key=lambda kv:abs(kv[1]),reverse=True))
        reg_models[label]={
            'chosen_alpha':alpha,
            'validation_mae':float(best[0]),
            'test':regression_eval(y[test],pt),
            'standardized_coefficients':coefs,
        }
        if label=='shape_plus_exposure': pred_count=pt

    # Baseline: previous 20 draw median.
    base=Xi[test][:,names.index('prev20_count_median')]
    results['regression']={'baseline_prev20_median':regression_eval(y[test],base),**reg_models}

    # Binary main crowd layer: 16+ vs <=15. Fit on train; choose C on validation AUC.
    Cs=[0.05,0.1,0.3,1,3,10]
    sc=StandardScaler().fit(Xi[train])
    Xtr=sc.transform(Xi[train]); Xv=sc.transform(Xi[val]); Xt=sc.transform(Xi[test])
    ybin=(y>=16).astype(int)
    best=None
    for C in Cs:
        clf=LogisticRegression(C=C,max_iter=5000,class_weight='balanced').fit(Xtr,ybin[train])
        pv=clf.predict_proba(Xv)[:,1]
        auc=roc_auc_score(ybin[val],pv)
        if best is None or auc>best[0]: best=(auc,C,clf)
    _,C,clf=best
    pp=clf.predict_proba(Xt)[:,1]; ph=(pp>=0.5).astype(int)
    coefs=dict(sorted(zip(names,clf.coef_[0].tolist()),key=lambda kv:abs(kv[1]),reverse=True))
    results['binary_16plus']={
        'definition':'1 if first-prize count >=16, else 0',
        'chosen_C':C,'validation_auc':float(best[0]),
        'test_auc':float(roc_auc_score(ybin[test],pp)),
        'test_accuracy':float(accuracy_score(ybin[test],ph)),
        'test_balanced_accuracy':float(balanced_accuracy_score(ybin[test],ph)),
        'standardized_coefficients':coefs,
        'confusion_matrix':confusion_matrix(ybin[test],ph).tolist(),
    }

    # Cleaner ② vs ③ comparison only: 6-15 vs16-49.
    mask_train=train[(y[train]>=6)&(y[train]<=49)]
    mask_val=val[(y[val]>=6)&(y[val]<=49)]
    mask_test=test[(y[test]>=6)&(y[test]<=49)]
    y23=(y>=16).astype(int)
    sc23=StandardScaler().fit(Xi[mask_train])
    A=sc23.transform(Xi[mask_train]); B=sc23.transform(Xi[mask_val]); Cx=sc23.transform(Xi[mask_test])
    best23=None
    for c in Cs:
        q=LogisticRegression(C=c,max_iter=5000,class_weight='balanced').fit(A,y23[mask_train])
        pv=q.predict_proba(B)[:,1]
        auc=roc_auc_score(y23[mask_val],pv)
        if best23 is None or auc>best23[0]: best23=(auc,c,q)
    _,c23,q=best23
    p23=q.predict_proba(Cx)[:,1]; h23=(p23>=.5).astype(int)
    coef23=dict(sorted(zip(names,q.coef_[0].tolist()),key=lambda kv:abs(kv[1]),reverse=True))
    results['layer2_vs3']={
        'definition':'Among counts 6-49: ②=6-15 vs ③=16-49',
        'test_n':int(len(mask_test)),'chosen_C':c23,'validation_auc':float(best23[0]),
        'test_auc':float(roc_auc_score(y23[mask_test],p23)),
        'test_accuracy':float(accuracy_score(y23[mask_test],h23)),
        'test_balanced_accuracy':float(balanced_accuracy_score(y23[mask_test],h23)),
        'standardized_coefficients':coef23,
    }

    # Spearman correlations on fixed test for raw features vs count.
    corr={}
    for j,n in enumerate(names):
        rho,p=spearmanr(Xi[test][:,j],y[test])
        corr[n]={'rho':float(rho),'p':float(p)}
    results['test_feature_spearman']=dict(sorted(corr.items(),key=lambda kv:abs(kv[1]['rho']),reverse=True))

    # Calibration-like terciles of predicted count on fixed test.
    q1,q2=np.quantile(pred_count,[1/3,2/3])
    buckets=[]
    for label,mask in [
        ('low',pred_count<=q1),('mid',(pred_count>q1)&(pred_count<=q2)),('high',pred_count>q2)
    ]:
        yy=y[test][mask]
        buckets.append({
            'bucket':label,'n':int(mask.sum()),'pred_median':float(np.median(pred_count[mask])),
            'actual_median':float(np.median(yy)),'actual_mean':float(np.mean(yy)),
            'share_16plus':float(np.mean(yy>=16)),
        })
    results['test_predicted_count_terciles']=buckets

    # Per-draw fixed-test CSV for audit.
    with OUT_CSV.open('w',encoding='utf-8') as f:
        f.write('draw,date,numbers,actual_count,actual_layer,abcd,band,predicted_count,p16plus\n')
        for k,i in enumerate(test):
            m=metas[i]
            f.write(','.join([
                str(m['draw']),m['date'],'-'.join(f'{n:02d}' for n in m['nums']),str(m['count']),m['count_layer'],m['abcd'],m['band'],
                f'{pred_count[k]:.4f}',f'{pp[k]:.6f}'
            ])+'\n')

    OUT_JSON.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({
        'regression_test':results['regression']['shape_plus_exposure']['test'],
        'binary16_test_auc':results['binary_16plus']['test_auc'],
        'layer2vs3_test_auc':results['layer2_vs3']['test_auc'],
        'terciles':buckets,
    },ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
