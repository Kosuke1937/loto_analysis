# Mini Loto Satellite Repair Add-on10 Validation

Date: 2026-10-20
Status: Research branch only. Main10 remains unchanged.

## Purpose
Keep the current Main10 philosophy intact and test a separate 10-ticket repair lane for the specific failure mode:
"strong 4-core found, but the 5th satellite is weakly connected and missed."

This is NOT a replacement for the Main10.

## Candidate families tested
All features use only data available before each draw.
Core score:
- 4-number core ranked over all 31C4 = 31,465 cores
- score = PairSum + 5 * TripleSum, using rolling last 300 draws

For each selected core, the 27 possible fifth numbers are ranked by incremental Pair/Triple connection.

Tested 10-ticket allocations:
- 1x10: top 1 core x weakest 10 satellites
- 2x5: top 2 cores x weakest 5 satellites
- 5x2: top 5 cores x weakest 2 satellites
- 10x1: top 10 cores x weakest 1 satellite
- 3x433: core ranks 1/2/3 x weakest 4/3/3 satellites

## Backtest

### Development: draws 1000-1199 (200 draws)
- 1x10: 3+ = 18, 4+ = 2, 5 = 0
- 2x5: 3+ = 25, 4+ = 2, 5 = 0
- 5x2: 3+ = 29, 4+ = 2, 5 = 0
- 10x1: 3+ = 40, 4+ = 5, 5 = 0
- 3x433: 3+ = 24, 4+ = 2, 5 = 0

### Fixed validation: draws 1200-1399 (200 draws)
- 1x10: 3+ = 23, 4+ = 1, 5 = 0
- 2x5: 3+ = 21, 4+ = 1, 5 = 0
- 5x2: 3+ = 25, 4+ = 1, 5 = 0
- 10x1: 3+ = 23, 4+ = 0, 5 = 0
- 3x433: 3+ = 23, 4+ = 1, 5 = 0

The 10x1 development result did not generalize and is not adopted.

### Stress check: draws 1400-1409 (10 draws)
5x2:
- 3+ = 4
- 4+ = 2
- 5 = 1
- Draw 1406 best = 4/5
- Draw 1409 = exact 5/5

Important: draw 1409 motivated this research family, so the 1409 exact hit is a mechanism demonstration, NOT unbiased proof of future performance.

## Draw 1409 reconstruction with 5x2
Top 5 cores, two weakest satellites each:

1. core #1 11-14-30-31 + 15 -> 11-14-15-30-31 (3/5)
2. core #1 11-14-30-31 + 16 -> 11-14-16-30-31 (3/5)
3. core #2 14-21-30-31 + 22 -> 14-21-22-30-31 (5/5 EXACT)
4. core #2 14-21-30-31 + 15 -> 14-15-21-30-31 (4/5)
5. core #3 14-27-30-31 + 09 -> 09-14-27-30-31 (3/5)
6. core #3 14-27-30-31 + 08 -> 08-14-27-30-31 (3/5)
7. core #4 03-14-30-31 + 25 -> 03-14-25-30-31 (3/5)
8. core #4 03-14-30-31 + 08 -> 03-08-14-30-31 (3/5)
9. core #5 07-14-30-31 + 25 -> 07-14-25-30-31 (3/5)
10. core #5 07-14-30-31 + 16 -> 07-14-16-30-31 (3/5)

## Interpretation
- This add-on is intentionally concentrated and therefore has poorer broad 3+ coverage than a diversified random-like 10-ticket portfolio.
- Its role is not broad coverage. Its role is tail repair when a very strong 4-core exists.
- 5x2 is preferred over 2x5 provisionally because it spreads risk across five core hypotheses while retaining the weak-satellite concept.
- No historical exact hit was observed in the 400-draw development+validation set, so this cannot yet be called proven.
- Draw 1409 is the first clear exact mechanism demonstration in the stress window.

## Current recommendation
Maintain:
- Main10 = canonical Flow/Band/Core+Satellite/Assembly/Coverage template
- Add-on10 Research = top 5 4-cores x 2 weakest satellites each

Do not merge Add-on10 into Main10 yet.
Forward-test from draw 1410 onward.
Track:
- Main10 best match
- Add-on10 best match
- combined 20-ticket best match
- incremental 3+/4+/5 gains from Add-on10
- overlap between Main10 and Add-on10
- whether Add-on10 exact/4+ gains persist
