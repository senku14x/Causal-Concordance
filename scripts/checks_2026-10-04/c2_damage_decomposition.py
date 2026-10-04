"""C2 (pre-registered in docs/design_specs/checks_2026-10-04_no_gpu.md): where the off-pair mass goes after the
whole-cell ':' patch. (A) 004 raw rows, arm colon|full, L42, 33 pairs: exact 12-colour full-sequence + next-token
masses. (B) 001 raw rows, emit_colon p->m, L40-60, 82 pairs: off-pair next-token mass + top-5 (partial). CPU only."""
import ast, glob, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/checks_2026-10-04/c2_damage_decomposition.json"
COLOURS = ("black", "blue", "brown", "gold", "gray", "green", "orange", "pink", "purple", "red", "teal", "white")
rng = np.random.default_rng(20261004)
CATS = ("a", "a2", "z", "z2", "other_colour", "non_colour")


def boot_median(x, n=10000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None]
    b = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def cat_of(colour, roles):
    for k in ("a", "a2", "z", "z2"):   # priority a > a' > z > z'
        if roles[k] == colour:
            return k
    return "other_colour"


def masses(prob_by_colour, roles):
    m = dict.fromkeys(CATS, 0.0)
    for c, p in prob_by_colour.items():
        m[cat_of(c, roles)] += p
    m["non_colour"] = 1.0 - sum(prob_by_colour.values())
    return m


def summarize(deltas):
    """deltas: list of {cat: delta}; share = delta_cat / off-pair increase over pairs with off-pair increase > 0.01."""
    out = {}
    off = np.array([sum(d[c] for c in ("z", "z2", "other_colour", "non_colour")) for d in deltas])
    keep = off > 0.01
    out["n_pairs"] = len(deltas); out["n_offpair_up_gt_0.01"] = int(keep.sum())
    out["offpair_increase_median"] = float(np.median(off)); out["offpair_increase_ci"] = boot_median(off)
    for c in CATS:
        x = np.array([d[c] for d in deltas])
        out[f"delta_{c}_median"] = float(np.median(x)); out[f"delta_{c}_ci"] = boot_median(x)
        out[f"delta_{c}_pos"] = int((x > 0).sum())
        if c not in ("a", "a2") and keep.any():
            sh = x[keep] / off[keep]
            out[f"share_{c}_median"] = float(np.median(sh)); out[f"share_{c}_ci"] = boot_median(sh)
    if keep.any():
        zz = np.array([(d["z"] + d["z2"]) for d in deltas])[keep] / off[keep]
        out["share_z_plus_z2_median"] = float(np.median(zz)); out["share_z_plus_z2_ci"] = boot_median(zz)
    return out


res = {"design": "docs/design_specs/checks_2026-10-04_no_gpu.md#c2"}

# ---------------- (A) 004, exact, L42 ----------------
A_full, A_next, A_pairs = [], [], []
for f in sorted(glob.glob(str(ROOT / "outputs/004_span_assay/run/pairs/*.jsonl"))):
    rows = [json.loads(l) for l in open(f)]
    clean = [r for r in rows if r.get("kind") == "clean"]
    full = [r for r in rows if r.get("kind") == "swap" and r.get("site") == "colon" and r.get("arm") == "full"]
    assert len(clean) == 1 and len(full) == 1, (f, len(clean), len(full))
    cl, ed = clean[0], full[0]
    mem = ast.literal_eval(cl["member"]) if isinstance(cl["member"], str) else cl["member"]
    edi = ast.literal_eval(ed["edit"]) if isinstance(ed["edit"], str) else ed["edit"]
    roles = {"a": cl["a"], "a2": cl["a2"], "z": cl["z"], "z2": cl["z2"]}
    pq0 = {c: float(np.exp(q)) for c, q in zip(COLOURS, mem["logq"])}
    pq1 = {c: float(np.exp(q)) for c, q in zip(COLOURS, edi["logq"])}
    m0, m1 = masses(pq0, roles), masses(pq1, roles)
    A_full.append({c: m1[c] - m0[c] for c in CATS})
    n0, n1 = mem.get("next_token_mass"), edi.get("next_token_mass")
    if n0 and n1:
        k0, k1 = masses(n0, roles), masses(n1, roles)
        A_next.append({c: k1[c] - k0[c] for c in CATS})
    A_pairs.append({"pair_id": cl["pair_id"], "S": ed["S"], "roles": roles, "full_clean": m0, "full_edit": m1,
                    "z_eq_a": roles["z"] == roles["a"]})
res["A_004_L42_full_sequence"] = summarize(A_full)
res["A_004_L42_next_token"] = summarize(A_next) if A_next else None
res["A_pairs"] = A_pairs

# ---------------- (B) 001, partial, all layers ----------------
pairs = {}
for l in open(ROOT / "outputs/001_leverage_profile/run/pairs_brew.jsonl"):
    r = json.loads(l)
    if r["direction"] == "p->m":
        pairs[r["pair_id"]] = r
# roles for 001 pairs from the gate record (member a/z, partner a'/z')
gate = {}
for l in open(ROOT / "results/001_leverage_profile/analysis/gate_pairs.jsonl"):
    g = json.loads(l); gate[g.get("pair_id")] = g
res["B_gate_record_fields"] = sorted(next(iter(gate.values())).keys())[:40]


def tok_cat(tok, roles):
    t = tok.strip().lower()
    if t in COLOURS:
        return cat_of(t, roles)
    return "non_colour"


B = {}
missing_roles = 0
for l in open(ROOT / "outputs/001_leverage_profile/run/profile_brew.jsonl"):
    r = json.loads(l)
    if r["region"] != "emit_colon" or r["direction"] != "p->m":
        continue
    p = pairs[r["pair_id"]]
    g = gate.get(r["pair_id"], {})
    roles = {"a": p["a"], "a2": p["a_donor"], "z": g.get("z") or g.get("member_z"), "z2": g.get("z_p")}
    if roles["z"] is None or roles["z2"] is None:
        missing_roles += 1
    vis0 = dict.fromkeys(CATS, 0.0); vis1 = dict.fromkeys(CATS, 0.0)
    for tok, lp in p["top5_clean"]:
        vis0[tok_cat(tok, roles)] += float(np.exp(lp))
    for tok, lp in r["top5"]:
        vis1[tok_cat(tok, roles)] += float(np.exp(lp))
    dv = {c: vis1[c] - vis0[c] for c in CATS}
    off = r["off_pair_mass_edit"] - r["off_pair_mass_clean"]
    vis_off = sum(dv[c] for c in ("z", "z2", "other_colour", "non_colour"))
    B.setdefault(r["layer"], []).append({"d": dv, "off": off, "unattributed": off - vis_off})
res["B_missing_roles"] = missing_roles
res["B_001_emit_colon_top5"] = {}
for L, rows in sorted(B.items()):
    off = np.array([x["off"] for x in rows]); un = np.array([x["unattributed"] for x in rows])
    d = {c: np.array([x["d"][c] for x in rows]) for c in CATS}
    res["B_001_emit_colon_top5"][L] = {
        "n": len(rows), "offpair_increase_median": float(np.median(off)), "frac_offpair_up_gt_0.10": float(np.mean(off > 0.10)),
        **{f"visible_delta_{c}_median": float(np.median(d[c])) for c in CATS},
        "visible_delta_z_plus_z2_median": float(np.median(d["z"] + d["z2"])),
        "unattributed_median": float(np.median(un))}

OUT.write_text(json.dumps(res, indent=1, default=float))
A = res["A_004_L42_full_sequence"]
print("(A) 004 ':' L42 full-sequence, n =", A["n_pairs"], "| off-pair up > 0.01:", A["n_offpair_up_gt_0.01"])
print("    off-pair increase median %.3f %s" % (A["offpair_increase_median"], [round(x, 3) for x in A["offpair_increase_ci"]]))
for c in CATS:
    s = A.get(f"share_{c}_median")
    print(f"    {c:12s} delta median {A[f'delta_{c}_median']:+.4f} {[round(x,4) for x in A[f'delta_{c}_ci']]} pos {A[f'delta_{c}_pos']}/{A['n_pairs']}"
          + (f" | share of off-pair increase {s:.2f} {[round(x,2) for x in A[f'share_{c}_ci']]}" if s is not None else ""))
print("    share z+z2 median %.2f %s" % (A["share_z_plus_z2_median"], [round(x, 2) for x in A["share_z_plus_z2_ci"]]))
if res["A_004_L42_next_token"]:
    N = res["A_004_L42_next_token"]
    print("(A') next-token: off-pair median %.3f; shares z %.2f z2 %.2f other %.2f non-colour %.2f; z+z2 %.2f" % (
        N["offpair_increase_median"], N["share_z_median"], N["share_z2_median"], N["share_other_colour_median"],
        N["share_non_colour_median"], N["share_z_plus_z2_median"]))
print("(B) 001 top-5, missing roles:", missing_roles)
for L, s in res["B_001_emit_colon_top5"].items():
    print(f"    L{L}: n={s['n']} off-pair up median {s['offpair_increase_median']:.3f} (frac>0.10 {s['frac_offpair_up_gt_0.10']:.2f}) | visible: "
          + " ".join(f"{c} {s[f'visible_delta_{c}_median']:+.3f}" for c in CATS) + f" | unattributed {s['unattributed_median']:+.3f}")
