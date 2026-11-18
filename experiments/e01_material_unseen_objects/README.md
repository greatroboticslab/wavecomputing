# E01: material of unseen objects from laser readings

Can the 3DAI laser readings name an object's material when the model has never seen that object, and do they do better
than a plain photo of the same object? The plan, with its predictions, is entry (a) of the
[lab notebook](../../EXPERIMENT_LOG.md).

## Run

From this folder (Python 3.10 or newer; `pip install -r ../../requirements.txt`):

```bash
python fetch_3dai.py --out ../../data/3dai                       # pinned 3DAI files + MANIFEST.tsv with SHA-256
D=../../data/3dai/6556bec6898e/dataset
python photo_color_stats.py --images $D/images --out ../../data/e01/photo_color_stats.csv
# on a GPU node with torch + transformers (Hamilton: the AGAI vision environment)
python embed_siglip2.py --images $D/images --model <siglip2-so400m-patch16-512> --out ../../data/e01/siglip2_embeddings.npz
python run_e01.py --features $D/laser_features.csv --a1 ../../data/e01/photo_color_stats.csv \
    --a2 ../../data/e01/siglip2_embeddings.npz --out ../../results/e01
```

`python run_e01.py --synthetic --out <dir>` runs the whole analysis on made-up data with a known answer.

## Outputs

`results/e01/summary.json` (every number), `report.md` (tables and the verdict on each prediction) and
`predictions_scans.csv` (per-scan predictions of the two main models).
