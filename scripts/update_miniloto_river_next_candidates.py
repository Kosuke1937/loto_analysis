from pathlib import Path
p=Path('miniloto-river.html')
s=p.read_text(encoding='utf-8')
old='  <div class="riverWrap"><div id="river" class="river"></div></div>\n  <div id="selectedBox" class="selectedBox"></div>'
new='  <div class="riverWrap"><div id="river" class="river"></div></div>\n  <div id="selectedBox" class="selectedBox"></div>\n  <div id="nextConditionBox" class="selectedBox"></div>'
if old not in s: raise SystemExit('html target not found')
s=s.replace(old,new,1)
old='function setNumber(n){selectedNumber=selectedNumber===n?null:n;render()}'
new='''function nextRelationMatches(n){
  const prev=DATA.length?mains(DATA.at(-1)):[];
  const prev2=DATA.length>1?mains(DATA.at(-2)):[];
  const prev35=[];for(let k=3;k<=5;k++)if(DATA.length>=k)prev35.push(...mains(DATA.at(-k)));
  const prevSame=prev.includes(n);
  const prevNear=prev.some(x=>Math.abs(n-x)===1);
  const prev2Near=prev2.some(x=>Math.abs(n-x)<=1);
  const prev35Same=prev35.includes(n);
  return {prevSame,prevNear,prev2Near,prev35Same,outside:!(prevSame||prevNear||prev2Near||prev35Same)};
}
function setNumber(n){selectedNumber=selectedNumber===n?null:n;render()}'''
if old not in s: raise SystemExit('function target not found')
s=s.replace(old,new,1)
old='''  const ny=T+rs.length*rowH+nextH/2;a.push(`<text class="drawLabel" x="${L-9}" y="${ny+4}" text-anchor="end">次回</text>`);for(let n=1;n<=31;n++){const xx=x(n);a.push(`<circle cx="${xx}" cy="${ny}" r="7" class="nextDot"/>`);if(n%5===0||n===1||n===31)a.push(`<text class="nextTxt" x="${xx}" y="${ny+3}" text-anchor="middle">${n}</text>`)}
  $('river').innerHTML=`<svg viewBox="0 0 ${W} ${H}" aria-label="ミニロト数字の川">${a.join('')}</svg>`;'''
new='''  const ny=T+rs.length*rowH+nextH/2;a.push(`<text class="drawLabel" x="${L-9}" y="${ny+4}" text-anchor="end">次回</text>`);for(let n=1;n<=31;n++){const xx=x(n),rel=nextRelationMatches(n);
    if(relationOn.outside&&rel.outside)a.push(`<circle cx="${xx}" cy="${ny}" r="16" class="relationHalo outside"/>`);
    if(relationOn.prev35Same&&rel.prev35Same)a.push(`<circle cx="${xx}" cy="${ny}" r="14.5" class="relationHalo prev35Same"/>`);
    if(relationOn.prev2Near&&rel.prev2Near)a.push(`<circle cx="${xx}" cy="${ny}" r="13" class="relationHalo prev2Near"/>`);
    if(relationOn.prevNear&&rel.prevNear)a.push(`<circle cx="${xx}" cy="${ny}" r="10.5" class="relationHalo prevNear"/>`);
    if(relationOn.prevSame&&rel.prevSame)a.push(`<circle cx="${xx}" cy="${ny}" r="9" class="relationHalo prevSame"/>`);
    a.push(`<circle data-n="${n}" cx="${xx}" cy="${ny}" r="7" class="nextDot" style="cursor:pointer"/>`);if(n%5===0||n===1||n===31)a.push(`<text class="nextTxt" x="${xx}" y="${ny+3}" text-anchor="middle">${n}</text>`)}
  $('river').innerHTML=`<svg viewBox="0 0 ${W} ${H}" aria-label="ミニロト数字の川">${a.join('')}</svg>`;'''
if old not in s: raise SystemExit('next row target not found')
s=s.replace(old,new,1)
old='function render(){const rs=rows();renderKpis(rs);renderRiver(rs);renderLists(rs)}'
new='''function renderNextConditionBox(){
  const inside=[],outside=[];for(let n=1;n<=31;n++){const m=nextRelationMatches(n);(m.outside?outside:inside).push(n)}
  $('nextConditionBox').innerHTML=`<b>次回の4条件Union ${inside.length}数字</b><br>${inside.map(pad).join('・')}<br><b>4条件外 ${outside.length}数字</b><br>${outside.map(pad).join('・')}<br><span class="label">※ 条件外は除外候補ではなく、現在の4つの流れ条件では説明できない数字として別管理します。</span>`;
}
function render(){const rs=rows();renderKpis(rs);renderRiver(rs);renderLists(rs);renderNextConditionBox()}'''
if old not in s: raise SystemExit('render target not found')
s=s.replace(old,new,1)
old='  <div class="helper">横＝数字1〜31、縦＝抽せん回。最新回は赤。数字見出しまたは丸をタップすると、その数字の列を強調します。</div>'
new='  <div class="helper">横＝数字1〜31、縦＝抽せん回。最新回は赤。「次回」行にもON中の条件リングを表示します。数字見出しまたは丸をタップすると、その数字の列を強調します。</div>'
if old in s:s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('updated',p)
