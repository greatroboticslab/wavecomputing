"""E01 baseline A2: image embeddings of each scan's plain photo from pretrained SigLIP 2 SO400M/16-512.

This is the backbone of AGAI v2's classifier, used as released (no strawberry fine-tuning). The processor resizes
each photo to 512x512; the embedding is the model's pooled image feature (1152 values).

    python embed_siglip2.py --images <commit dir>/dataset/images --model <path to siglip2-so400m-patch16-512> \
        --out siglip2_embeddings.npz

Output: names (folder <object>__<pose>__<scan8>) and emb (float32, n x d).
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(a.model).float().eval().to(dev)  # float32 on any transformers version
    proc = AutoProcessor.from_pretrained(a.model)
    paths = sorted(glob.glob(os.path.join(a.images, "*", "*", "color.jpg")))
    names, embs = [], []
    for i in range(0, len(paths), a.batch):
        chunk = paths[i:i + a.batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        inputs = proc(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        if not torch.is_tensor(out):  # newer transformers return a model output
            out = out.pooler_output
        embs.append(out.float().cpu().numpy())
        names += [os.path.basename(os.path.dirname(p)) for p in chunk]
    emb = np.concatenate(embs)
    np.savez(a.out, names=np.array(names), emb=emb.astype(np.float32))
    print(f"{len(names)} photos, dim {emb.shape[1]}, device {dev} -> {a.out}")


if __name__ == "__main__":
    main()
