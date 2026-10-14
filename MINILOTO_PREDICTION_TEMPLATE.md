# ミニロト候補作成 正本テンプレ

更新日: 2026-10-15

## 目的
毎回の候補生成で、1406回の Core+Satellite と1407回の Flow×Band Joint State の成功要因を残しつつ、1408回で発覚した Flow偏重・Bandとの混同・未観測Cell排除を防ぐ。

---

## 1. 用語を分離
### Flow
- F-A: 前回本数字そのもの
- F-B: 前回本数字±1
- F-C: 2回前の同一・±1
- F-D: 3〜5回前本数字
- F-O: 上記以外

優先順位は F-A → F-B → F-C → F-D → F-O。排他的に分類する。

### Band
数字帯は 1–9 / 10–19 / 20–29 / 30–31。
頻度クラスは Band-A / Band-B / Band-C / Band-D と表記し、Flowと絶対に混同しない。

---

## 2. 候補空間
- 母集団は31C5=169,911通りから開始。
- 合計、帯、Flow、Joint State、連番、30/31有無、20番台有無などを原則ハード除外しない。
- 過去0回のJoint Stateも除外しない。
- 「過去に少ない」と「数学的に成立しない」を区別する。

---

## 3. 成功系統
### 1406回
Core+Satelliteを残す。
- 強いPair/Triple Core
- 4-core
- CoreへのSatellite接続
- 5th候補の複線化

### 1407回
Flow×Band Joint State + Assemblyを残す。
- 優先Flow
- 粗Flow = F-A数-(F-B+F-C)数-F-D数
- Band構成
- Joint State
- Cell内Assembly

ただし1407の具体的Stateを次回へコピーしない。

---

## 4. Joint Stateの扱い
Joint Stateはハードフィルターにしない。
評価は
- Flow単独
- Band単独
- Joint State補正
のsoft scoringとする。

未観測Joint Cellは候補資格を維持する。

---

## 5. Assembly
構造評価の基礎は
PairSum + 5 × TripleSum

これは1407購入#4のCell内順位を記録値とほぼ再現できたため採用。
ただし「Assembly順位=当選確率」とは解釈しない。

---

## 6. Core+Satellite
- 3-core、4-coreを確認
- SatelliteはCoreへのPair接続で評価
- 強い4-coreがある場合、5th違いを最大1〜2口まで許容
- Rescueは抽選前ルールとして実装し、結果後の後付けは禁止

---

## 7. Flow Coverage監査
10口確定前に必ずUnion内のFlow別Coverageを表示する。

最低監査目安:
- F-A: 50%以上
- F-B: 50%以上
- F-C: 50%以上
- F-D: 50%以上
- F-O: 33%以上

一つのFlowだけ極端に低ければ候補確定禁止。

---

## 8. 各口Flow監査
10口すべてについて
- 候補数字
- F-A/F-B/F-C/F-D/F-O個数
- 粗Flow
を表示する。

特定Flow構成への過集中を検出する。

---

## 9. Band分布
Band-A/B/C/Dの配分を表示する。
過去の4:3:2:1は参考値であり固定ルールではない。

---

## 10. ポートフォリオ制約
標準10口:
- 同一数字: 原則最大4口
- 同一Pair: 最大2口程度
- 同一Triple: 最大1口程度
- Union: おおむね20〜24数字
- 3連番: 0〜1口
- 2連番: 許容
- 合計値: ハード除外しない
- Novel Joint Cell: 排除しない

---

## 11. 最終STOPチェック
以下が全てOKになるまで最終10口を出さない。

- [ ] F-A/B/C/D/Oを正しく排他的分類
- [ ] FlowとBand層を混同していない
- [ ] 1406 Core+Satelliteを確認
- [ ] 1407 Joint State思想を確認
- [ ] Joint Stateをハード除外にしていない
- [ ] 未観測Cellを切っていない
- [ ] Flow別Union Coverageを確認
- [ ] 特定Flowをほぼ消していない
- [ ] 数字最大4口程度
- [ ] Pair最大2口程度
- [ ] Triple最大1口程度
- [ ] 3連番0〜1口
- [ ] 合計値をハード条件にしていない
- [ ] Committeeを候補生成に使っていない
- [ ] 結果後情報を使っていない

1項目でもNGなら再Assemblyする。

---

## 12. 毎回の最終出力
### Flow数字群
F-A:
F-B:
F-C:
F-D:
F-O:

### 10口
| # | 数字 | 粗Flow | Band | 合計 | Assembly | 役割 |
|---:|---|---|---|---:|---:|---|

### ポートフォリオ監査
- Union: __/31
- F-A Coverage: __/__
- F-B Coverage: __/__
- F-C Coverage: __/__
- F-D Coverage: __/__
- F-O Coverage: __/__
- 最大数字使用: __口
- 最大Pair: __口
- 最大Triple: __口
- 3連番: __口
- Novel Joint Cell: __口

### 判定
- OK: 最終候補として採用
- NG: Coverage/偏り異常のため再Assembly

---

## 最上位原則
「前回当たった形」をコピーしない。
「前回当たった生成思想」を、今回の最新状態で再計算・再監査する。
