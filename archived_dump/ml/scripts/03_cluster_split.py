"""Step 3: group near-duplicate images and make a leakage-free split.

- Near-duplicates (perceptual hash distance <= max_hamming) share a cluster_id.
- Train-pool sources are split train/val/test by cluster, stratified by label,
  so near-duplicates never land in different splits.
- External test sources become split "test_<source>". Train-pool images that are
  near-duplicates of an external test image are marked "excluded_leak".
- Classes above max_per_class are capped (whole clusters dropped -> "excluded_cap").
Updates data/metadata.csv in place.
"""
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import SPLITS, die, load_config, resolve_path  # noqa: E402


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_hashes(hashes: list[str], n_bits: int, max_dist: int) -> np.ndarray:
    """Clusters hashes so that any pair within max_dist bits ends up together.

    Uses band bucketing instead of comparing all pairs: split each hash into
    max_dist + 1 bands; two hashes differing in <= max_dist bits must agree
    exactly on at least one band, so only same-bucket pairs need checking.
    """
    values = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    uf = UnionFind(len(values))
    n_bands = max_dist + 1
    band_bits = math.ceil(n_bits / n_bands)
    for b in range(n_bands):
        keys = (values >> np.uint64(b * band_bits)) & np.uint64((1 << band_bits) - 1)
        buckets = defaultdict(list)
        for i, k in enumerate(keys.tolist()):
            buckets[k].append(i)
        for idx in buckets.values():
            if len(idx) < 2:
                continue
            idx = np.array(idx)
            vals = values[idx]
            for a in range(len(idx) - 1):
                dist = np.bitwise_count(vals[a] ^ vals[a + 1:])
                for j in np.nonzero(dist <= max_dist)[0]:
                    uf.union(int(idx[a]), int(idx[a + 1 + j]))
    roots = [uf.find(i) for i in range(len(values))]
    # renumber roots to 0..k-1 for readable ids
    remap = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def assign_splits(pool: pd.DataFrame, ratios: dict, max_per_class: int | None, seed: int) -> dict[int, str]:
    """Returns {cluster_id: split}. Each cluster is assigned by its majority label."""
    rng = np.random.default_rng(seed)
    sizes = pool.groupby("cluster_id").size().to_dict()
    majority = pool.groupby("cluster_id")["label"].agg(lambda s: s.value_counts().index[0]).to_dict()
    by_label = defaultdict(list)
    for cid, lab in majority.items():
        by_label[lab].append(cid)

    result = {}
    for lab in sorted(by_label):
        cids = sorted(by_label[lab])
        rng.shuffle(cids)
        if max_per_class:
            kept, total = [], 0
            for cid in cids:
                if total >= max_per_class:
                    result[cid] = "excluded_cap"
                    continue
                kept.append(cid)
                total += sizes[cid]
            cids = kept
        total = sum(sizes[c] for c in cids)
        filled = {s: 0 for s in ratios}
        # Largest clusters first, then give each cluster to the split furthest below its target.
        for cid in sorted(cids, key=lambda c: -sizes[c]):
            split = max(ratios, key=lambda s: ratios[s] * total - filled[s])
            result[cid] = split
            filled[split] += sizes[cid]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: configs/config.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scfg = cfg["split"]
    csv_path = resolve_path(cfg, "metadata_csv")
    if not csv_path.is_file():
        die(f"{csv_path} not found. Run 02_build_metadata.py first.")
    df = pd.read_csv(csv_path, dtype={"hash": str})
    if df.empty or df["hash"].isna().any():
        die(f"{csv_path} is empty or has rows without a hash. Re-run 02_build_metadata.py.")
    ratios = scfg["ratios"]
    if set(ratios) != set(SPLITS) or not math.isclose(sum(ratios.values()), 1.0):
        die(f"split.ratios must have keys {SPLITS} summing to 1.0, got {ratios}")

    print(f"Clustering {len(df)} hashes (max_hamming={scfg['max_hamming']})...")
    df["cluster_id"] = cluster_hashes(df["hash"].tolist(), scfg["hash_size"] ** 2, scfg["max_hamming"])
    sizes = df.groupby("cluster_id").size()
    print(f"{sizes.size} clusters; {int((sizes > 1).sum())} contain near-duplicates; largest has {sizes.max()} images")
    mixed = df.groupby("cluster_id")["label"].nunique()
    if (mixed > 1).any():
        print(f"WARNING: {int((mixed > 1).sum())} clusters mix labels (same-looking image, different label). "
              "Check them in metadata.csv; they are split by majority label.")

    roles = {s: v["role"] for s, v in cfg["sources"].items()}
    is_ext = df["source"].map(roles).eq("external_test")
    df["split"] = ""
    df.loc[is_ext, "split"] = "test_" + df.loc[is_ext, "source"]

    leak = ~is_ext & df["cluster_id"].isin(set(df.loc[is_ext, "cluster_id"]))
    df.loc[leak, "split"] = "excluded_leak"
    pool_mask = ~is_ext & ~leak
    if not pool_mask.any():
        die("No train-pool images left (check 'role: train' sources in config.yaml).")

    mapping = assign_splits(df[pool_mask], ratios, scfg.get("max_per_class"), scfg["seed"])
    df.loc[pool_mask, "split"] = df.loc[pool_mask, "cluster_id"].map(mapping)

    df.to_csv(csv_path, index=False)
    print(f"\nexcluded as leakage into external test: {int(leak.sum())}")
    print("\nImages per split/class:")
    print(df.pivot_table(index="split", columns="label", values="path", aggfunc="count", fill_value=0)
            .reindex(columns=cfg["classes"], fill_value=0).to_string())
    counts = Counter(df.loc[pool_mask, "split"])
    used = sum(counts[s] for s in SPLITS)
    if used:
        print("\nTrain-pool fractions: " + ", ".join(f"{s}={counts[s] / used:.3f}" for s in SPLITS))
    print(f"\nUpdated {csv_path}")


if __name__ == "__main__":
    main()
