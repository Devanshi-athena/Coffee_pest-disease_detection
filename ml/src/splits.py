"""Grouped, stratified train/val/test assignment (pandas + numpy only)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def assign_group_splits(groups: pd.Series, strata: pd.Series, ratios: dict, seed: int) -> pd.Series:
    """Assigns every row a split so that a group never straddles splits.

    Each group is assigned by its majority stratum; within a stratum, largest groups go first and each goes
    to the split that is furthest below its target share. Returns a Series aligned with `groups`.
    """
    df = pd.DataFrame({"g": groups.to_numpy(), "s": strata.to_numpy()})
    sizes = df.groupby("g").size()
    majority = df.groupby("g")["s"].agg(lambda s: s.value_counts().index[0])
    rng = np.random.default_rng(seed)
    by_stratum = defaultdict(list)
    for g, s in majority.items():
        by_stratum[s].append(g)
    result = {}
    for s in sorted(by_stratum):
        gs = sorted(by_stratum[s], key=str)
        rng.shuffle(gs)
        total = sum(sizes[g] for g in gs)
        filled = {k: 0 for k in ratios}
        for g in sorted(gs, key=lambda g: -sizes[g]):  # stable sort keeps the shuffled order for ties
            split = max(ratios, key=lambda k: ratios[k] * total - filled[k])
            result[g] = split
            filled[split] += sizes[g]
    return pd.Series(df["g"].map(result).to_numpy(), index=groups.index)
