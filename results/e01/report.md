# E01 results

Scans analysed: 79 of 80 (scans with laser features and a photo). Primary materials: cardboard, foam, metal, paper, plastic.
Laser features: 78; dropped for missingness: none.

## Leak-free (leave one object out), primary materials

| features / model | balanced acc. | 95% CI | macro F1 | object acc. | object bal. acc. |
|---|---|---|---|---|---|
| L/logreg | 0.249 | 0.146-0.431 | 0.248 | 0.516 | 0.304 |
| L/rf | 0.214 | 0.161-0.344 | 0.182 | 0.484 | 0.212 |
| A1/logreg | 0.033 | 0.000-0.100 | 0.022 | 0.000 | 0.000 |
| A1/rf | 0.184 | 0.121-0.294 | 0.176 | 0.419 | 0.188 |
| A2/logreg | 0.231 | 0.152-0.386 | 0.226 | 0.516 | 0.263 |
| A2/rf | 0.220 | 0.182-0.344 | 0.181 | 0.581 | 0.250 |
| L+A2/logreg | 0.237 | 0.157-0.396 | 0.234 | 0.548 | 0.275 |
| L+A2/rf | 0.183 | 0.163-0.282 | 0.127 | 0.516 | 0.200 |

Chance balanced accuracy: 0.200.

## Leaky split (5-fold by scan, 20 repeats)

| features / model | mean | sd |
|---|---|---|
| L/logreg | 0.466 | 0.038 |
| L/rf | 0.369 | 0.028 |
| A2/logreg | 0.778 | 0.045 |

## Laser against photo (logistic regression, leak-free)

| comparison | difference in balanced acc. | 95% CI | objects right only with laser / only with photo | McNemar p |
|---|---|---|---|---|
| L-A2/logreg | 0.018 | -0.118 to 0.169 | 3 / 3 | 1 |
| L-A1/logreg | 0.216 | 0.070 to 0.423 | 16 / 0 | 3.05e-05 |

Metal vs not metal (all objects, L/logreg): object accuracy 0.629 (95% CI 0.449-0.785), balanced accuracy 0.544.

## Predictions

- P1: L/logreg leak-free balanced accuracy: lower 95% bound > 0.20 (chance): value 0.146 -> NOT supported
- P2: leaky mean minus leak-free balanced accuracy >= 0.10 (L/logreg): value 0.217 -> supported
- P3: L/logreg beats A2/logreg on leak-free balanced accuracy (point estimate): value 0.018 -> supported
- P4: metal vs not metal, all objects, L/logreg leak-free object accuracy >= 0.85: value 0.629 -> NOT supported

## Exploratory

| analysis | balanced acc. | object acc. | n objects |
|---|---|---|---|
| all_materials/L/logreg | 0.110 | 0.400 | 35 |
| all_materials/A2/logreg | 0.144 | 0.457 | 35 |
| channel_1_only/logreg | 0.122 | 0.226 | 31 |
| channel_2_only/logreg | 0.178 | 0.290 | 31 |
| channel_3_only/logreg | 0.163 | 0.323 | 31 |
| channel_4_only/logreg | 0.159 | 0.194 | 31 |
| without_possibly_mislabelled/L/logreg | 0.188 | 0.483 | 29 |

Per-material recall, L/logreg leak-free: cardboard 0.000, foam 0.000, metal 0.368, paper 0.333, plastic 0.543.

Confusion matrix, L/logreg leak-free (rows true, columns predicted; order cardboard, foam, metal, paper, plastic): [[0, 0, 3, 0, 4], [1, 0, 1, 0, 2], [3, 0, 7, 2, 7], [1, 0, 3, 2, 0], [4, 2, 8, 2, 19]].
