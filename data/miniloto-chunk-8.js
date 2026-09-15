window.MINI_CHUNKS=window.MINI_CHUNKS||[];window.MINI_CHUNKS.push([[1401,"2026/08/25",15,17,23,26,31,22,17],[1402,"2026/09/01",1,4,20,25,29,22,7],[1403,"2026/09/08",8,11,14,17,19,24,null],[1404,"2026/09/15",2,3,4,9,31,11,null]]);
(function(){
  const all=(window.MINI_CHUNKS||[]).flat();
  const cur=all[all.length-1];
  if(!cur||cur[0]!==1404)return;
  const nums=r=>r.slice(2,7);
  const sum=a=>a.reduce((x,y)=>x+y,0);
  const band=a=>[a.filter(n=>n<=9).length,a.filter(n=>n>=10&&n<=19).length,a.filter(n=>n>=20&&n<=29).length,a.filter(n=>n>=30).length].join('-');
  const maxRun=a=>{let m=1,r=1;for(let i=1;i<a.length;i++){if(a[i]===a[i-1]+1){r++;if(r>m)m=r}else r=1}return m};
  const isStrict=a=>sum(a)<=49&&band(a)==='4-0-0-1'&&maxRun(a)>=3;
  const hist=all.slice(0,-1);
  const prevOf=pred=>{for(let i=hist.length-1;i>=0;i--){if(pred(nums(hist[i])))return hist[i]}return null};
  const cnt=pred=>hist.reduce((n,r)=>n+(pred(nums(r))?1:0),0);
  const strictPrev=prevOf(isStrict), strictCnt=cnt(isStrict);
  const lowPred=a=>sum(a)<=49, triPred=a=>maxRun(a)>=3, bandPred=a=>band(a)==='4-0-0-1';
  const lowPrev=prevOf(lowPred), triPrev=prevOf(triPred), bandPrev=prevOf(bandPred);
  const lowCnt=cnt(lowPred), triCnt=cnt(triPred), bandCnt=cnt(bandPred);
  let total=0,strictTheo=0,lowTheo=0,triTheo=0,bandTheo=0;
  for(let a=1;a<=27;a++)for(let b=a+1;b<=28;b++)for(let c=b+1;c<=29;c++)for(let d=c+1;d<=30;d++)for(let e=d+1;e<=31;e++){
    const x=[a,b,c,d,e]; total++;
    if(lowPred(x))lowTheo++;
    if(triPred(x))triTheo++;
    if(bandPred(x))bandTheo++;
    if(isStrict(x))strictTheo++;
  }
  const pct=(n,d)=>`${(100*n/d).toFixed(n/d<.001?3:2)}%`;
  const odds=(n,d)=>`約1/${Math.round(d/n).toLocaleString()}`;
  const fmtPrev=r=>r?`第${r[0]}回 ${nums(r).map(n=>String(n).padStart(2,'0')).join('・')}`:'過去なし（第1〜1403回）';
  const ver=document.querySelector('.version'); if(ver)ver.textContent='App 0.12.4';
  const foot=document.querySelector('.footer'); if(foot)foot.textContent='Loto Analysis App 0.12.4';
  const sub=document.querySelector('.sub'); if(sub)sub.textContent='第1404回まで反映。表示期間の抽選結果を動的集計。';
  const firstCard=sub&&sub.nextElementSibling&&sub.nextElementSibling.classList.contains('card')?sub.nextElementSibling:null;
  if(firstCard){
    const last3=all.slice(-3).map(r=>sum(nums(r)));
    firstCard.innerHTML=`<div class="label">直近の合計値遷移</div><div class="n">${last3.join(' → ')}</div><div class="small">第1404回は合計49。前回69から-20。合計レンジはハード除外ではなくsoft priorとして監視する。</div>`;
  }
  const card=document.createElement('div'); card.className='card'; card.style.marginBottom='10px';
  card.innerHTML=`<div class="label">第1404回 変化球監査</div><div class="n">02・03・04・09・31</div><div class="small" style="line-height:1.7">合計49｜帯構成4-0-0-1｜3連番02-03-04｜20番台0個<br><b>複合パターン</b>（合計≤49＋4-0-0-1＋3連番以上）：過去 ${strictCnt}回／1403回、直前は ${fmtPrev(strictPrev)}。理論 ${strictTheo}/${total} = ${pct(strictTheo,total)}（${odds(strictTheo,total)}）。<br><b>個別</b>：合計≤49 過去${lowCnt}回・直前 ${fmtPrev(lowPrev)}・理論${pct(lowTheo,total)}／3連番以上 過去${triCnt}回・直前 ${fmtPrev(triPrev)}・理論${pct(triTheo,total)}／帯4-0-0-1 過去${bandCnt}回・直前 ${fmtPrev(bandPrev)}・理論${pct(bandTheo,total)}。<br>今回の教訓：流れ条件②〜④で5数字を覆えていても、合計・帯・20番台必須・3連番禁止・特定数字除外をハードに重ねると正解を多重除外しうる。</div>`;
  if(firstCard)firstCard.insertAdjacentElement('afterend',card); else if(sub)sub.insertAdjacentElement('afterend',card);
})();
