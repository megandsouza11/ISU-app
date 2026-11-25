"""
Step 1 — run ONCE (locally or in Colab) to turn every product image into a CLIP embedding.

Usage:
    python embed.py --data data/fashion-product-images-small --limit 10000

Outputs (in ./artifacts):
    catalog.csv      product metadata for every embedded image
    embeddings.npy   one 512-d vector per product (float16, L2-normalized)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="folder containing styles.csv and images/")
parser.add_argument("--limit", type=int, default=None, help="embed a random subset (faster, smaller app)")
parser.add_argument("--out", default="artifacts")
parser.add_argument("--batch", type=int, default=64)
args = parser.parse_args()

data = Path(args.data)
out = Path(args.out)
out.mkdir(exist_ok=True)

# styles.csv has a few malformed rows (commas inside product names) -> skip them
styles = pd.read_csv(data / "styles.csv", on_bad_lines="skip")
styles["image_path"] = styles["id"].apply(lambda i: str(data / "images" / f"{i}.jpg"))
styles = styles[styles["image_path"].apply(lambda p: Path(p).exists())]
styles = styles.dropna(subset=["productDisplayName"])

if args.limit and args.limit < len(styles):
    styles = styles.sample(args.limit, random_state=42)
styles = styles.reset_index(drop=True)
print(f"Embedding {len(styles):,} products...")

# CLIP puts images AND text in the same vector space -> photo search and text search for free
model = SentenceTransformer("clip-ViT-B-32")

vectors = []
for start in tqdm(range(0, len(styles), args.batch)):
    paths = styles["image_path"].iloc[start:start + args.batch]
    images = [Image.open(p).convert("RGB") for p in paths]
    vectors.append(model.encode(images, normalize_embeddings=True, convert_to_numpy=True))

embeddings = np.vstack(vectors).astype("float16")
np.save(out / "embeddings.npy", embeddings)
styles.to_csv(out / "catalog.csv", index=False)
print(f"Saved {embeddings.shape} embeddings to {out}/")
