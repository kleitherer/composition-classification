import pandas as pd
import numpy as np
import itertools

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
df = pd.read_csv("./features/extracted_features.csv")

# For "feature validation between composers", it's cleaner to
# restrict to the big 5 where you have many pieces.
BIG_FIVE = ["bach", "mozart", "beethoven", "chopin", "debussy"]
df = df[df["composer"].isin(BIG_FIVE)]

composers = sorted(df["composer"].unique())
features = [c for c in df.columns if c != "composer"]

print("Composers used:", composers)
print("Num features:", len(features))


# ------------------------------------------------------------
# 2. Build PMFs with shared bins per feature
# ------------------------------------------------------------
def make_shared_bins(values_all, num_bins=20):
    """
    Create global bin edges for a feature across all composers.
    """
    vmin = np.min(values_all)
    vmax = np.max(values_all)
    if vmin == vmax:
        vmax = vmin + 1e-6
    return np.linspace(vmin, vmax, num_bins + 1)


def pmf_from_hist(values, bin_edges, eps=1e-6):
    """
    Histogram -> pmf with pseudocounts.
    """
    counts, _ = np.histogram(values, bins=bin_edges)
    counts = counts.astype(float) + eps
    pmf = counts / counts.sum()
    return pmf


# ------------------------------------------------------------
# 3. Symmetric KL between two PMFs
# ------------------------------------------------------------
def kl_pmf(p, q):
    """
    KL(P||Q) for discrete pmfs. Assumes no zeros (we added eps).
    """
    return np.sum(p * np.log(p / q))


def sym_kl(p, q):
    """
    Symmetrized KL: 0.5 * (KL(P||Q) + KL(Q||P)).
    Still >= 0, but less insane in one direction.
    """
    return 0.5 * (kl_pmf(p, q) + kl_pmf(q, p))


# ------------------------------------------------------------
# 4. Precompute PMFs per (composer, feature)
# ------------------------------------------------------------
pmfs = {}  # pmfs[(composer, feature)] = (pmf, bin_edges)

for feat in features:
    vals_all = df[feat].dropna().values
    bins = make_shared_bins(vals_all, num_bins=20)

    for comp in composers:
        vals = df[df["composer"] == comp][feat].dropna().values

        # if someone has almost no pieces, skip them for this feature
        if len(vals) < 8:
            continue

        pmf = pmf_from_hist(vals, bins)
        pmfs[(comp, feat)] = (pmf, bins)


# ------------------------------------------------------------
# 5. Compute symmetric KL for all composer pairs
# ------------------------------------------------------------
rows = []

for feat in features:
    # only consider composers that actually have a pmf for this feature
    comps_available = [c for c in composers if (c, feat) in pmfs]
    if len(comps_available) < 2:
        continue

    for cA, cB in itertools.combinations(comps_available, 2):
        p, _ = pmfs[(cA, feat)]
        q, _ = pmfs[(cB, feat)]

        d_sym = sym_kl(p, q)

        rows.append({
            "feature": feat,
            "composer_A": cA,
            "composer_B": cB,
            "KL_sym": d_sym
        })

kl_df = pd.DataFrame(rows)


# ------------------------------------------------------------
# 6. Nicely formatted summary for a given feature
# ------------------------------------------------------------
def summarize_feature(feature):
    sub = kl_df[kl_df["feature"] == feature]
    if sub.empty:
        print(f"\nNo data for feature {feature}")
        return

    # sort by symmetric KL
    sub_sorted = sub.sort_values("KL_sym")

    most_similar = sub_sorted.head(3)
    most_diff    = sub_sorted.tail(3)

    print("\n==============================")
    print(f"Symmetric KL Summary for Feature: {feature}")
    print("==============================\n")

    print("Most Similar (Lowest sym KL):")
    for _, row in most_similar.iterrows():
        print(f"  {row['composer_A']} ↔ {row['composer_B']}:  KL_sym = {row['KL_sym']:.3f}")

    print("\nMost Different (Highest sym KL):")
    for _, row in most_diff.iterrows():
        print(f"  {row['composer_A']} ↔ {row['composer_B']}:  KL_sym = {row['KL_sym']:.3f}")

    print("\n------------------------------------")


# ------------------------------------------------------------
# 7. Run for a few musically meaningful features
# ------------------------------------------------------------
for feat in ["interval_entropy", "leap_ratio", "accidental_ratio"]:
    summarize_feature(feat)