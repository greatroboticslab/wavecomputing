# E02 results: train on the 3DAI scanner, test on the robotic arm

Plan: lab notebook entry (f); results: entry (g). Data: 3DAI `6556bec` (red channels laser_ch1 and laser_ch2),
roboticsarmAI `96b76af` (primary photos). Training: 129 scanner spots of
29 objects ({'foam': 2, 'metal': 8, 'paper': 3, 'plastic': 16}). Test: 1084 arm spots in
30 object-parts ({'foam': 2, 'metal': 10, 'paper': 3, 'plastic': 15}).

Balanced accuracy over object-parts (a part's prediction = mean probability over its photos); bootstrap 95% interval
over parts; permutation test shuffling the scanner's materials among its objects.

| analysis | balanced acc. | 95% CI | chance | null mean | P (shuffles) | parts |
|---|---|---|---|---|---|---|
| primary: 4 materials, features standardized per rig | 0.267 | 0.170-0.431 | 0.25 | 0.263 | 0.48 (1000) | 30 |
| 4 materials, standardized on the scanner only | 0.250 | 0.250-0.333 | 0.25 | 0.259 | 0.54 (200) | 30 |
| plastic against metal, per rig | 0.517 | 0.306-0.717 | 0.50 | 0.468 | 0.35 (1000) | 25 |
| primary without the 7 flagged arm objects | 0.239 | 0.083-0.425 | 0.25 | 0.257 | 0.59 (200) | 20 |

References, leave one object out within one rig (same 16 features): scanner 0.224, arm
0.358 (post hoc permutation test, materials shuffled among the arm's parts: null mean
0.234, 95th percentile 0.475, P = 0.16, 200 shuffles).

Predictions: P1 (primary above chance) not supported; P2 (within the arm above the cross-rig primary) holds as a
number but is not significant; P3 (plastic against metal above chance) not supported.
