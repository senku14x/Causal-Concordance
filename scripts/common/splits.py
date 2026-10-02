"""Dev/test splits (design 001 §2). Fixed by seed before any outcome is seen.

brew:  by table (one table per item), 50/50, stratified by z; engineering items forced into dev.
chain: by step-sequence group (items with the same steps share prompts), among the usable single-digit-intermediate
       items, 50/50 of groups, stratified by depth; engineering items' groups forced into dev.
"""
import random
from collections import defaultdict

import partners as P

SEED = 20261002
BREW_FORCE_DEV = ("brew-s2-r10d2-0000", "brew-s2-r10d2-0001")
CHAIN_FORCE_DEV = ("chain-w2-d3p-0408",)


def _balanced(groups_by_stratum, seed, force_dev=()):
    """groups_by_stratum: {stratum: [group_id, ...]}. Returns {group_id: 'dev'|'test'}: within every stratum the groups
    are shuffled (seeded) and split in half; odd strata alternately round up and down (seeded starting parity), so the
    overall split is as close to 50/50 as possible. A forced-dev group displaces one dev group of its stratum to test."""
    rng = random.Random(seed)
    up = rng.randint(0, 1)
    out = {}
    for stratum in sorted(groups_by_stratum, key=str):
        gs = sorted(groups_by_stratum[stratum])
        rng.shuffle(gs)
        n_dev = len(gs) // 2
        if len(gs) % 2:
            n_dev += up
            up = 1 - up
        for i, g in enumerate(gs):
            out[g] = "dev" if i < n_dev else "test"
        for g in [g for g in gs if g in force_dev and out[g] == "test"]:
            swap = next((h for h in gs if out[h] == "dev" and h not in force_dev), None)
            out[g] = "dev"
            if swap is not None:
                out[swap] = "test"
    return out


def brew_split(items, seed=SEED, force_dev=BREW_FORCE_DEV):
    by_z = defaultdict(list)
    for it in items:
        by_z[it["intermediates"][0]].append(it["name"])
    return _balanced(by_z, seed, set(force_dev))


def chain_groups(items):
    """{group_key: [item names]} over usable items; group key = the step sequence."""
    g = defaultdict(list)
    for it in items:
        if P.chain_single_token_item(it):
            g[" | ".join(it["steps"])].append(it["name"])
    return dict(g)


def chain_split(items, seed=SEED, force_dev=CHAIN_FORCE_DEV):
    groups = chain_groups(items)
    by_depth = defaultdict(list)
    for key, names in groups.items():
        by_depth[len(key.split(" | "))].append(key)
    forced = {k for k, names in groups.items() if any(n in force_dev for n in names)}
    gsplit = _balanced(by_depth, seed, forced)
    return {n: gsplit[k] for k, names in groups.items() for n in names}, gsplit
