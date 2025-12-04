"""
CS109-STYLE BOOTSTRAP TEST:
Is the disputed harp sonata consistent with Dussek's authentic style?

For each feature:
  - Compute Dussek's empirical distribution
  - Compute how extreme the disputed value is
  - Use bootstrapping to estimate a p-value

Finally:
  - Combine evidence across all features
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_extraction import extract_features

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
EXTRACTED = "./features/extracted_features.csv"
DISPUTED_PATH = (
    "/Users/kaitlynleitherer/Desktop/CS109/final_project/data/"
    "instruments/harpists/dussek/sonata-en-do-menor-jan-ladislav-dussek.mid"
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(EXTRACTED)
df = df[df["composer"].notnull()]

dus = df[df["composer"] == "dussek"].copy()
other = df[df["composer"] != "dussek"].copy()

print("Dussek pieces:", len(dus))
if len(dus) < 6:
    print("WARNING: very small dataset — statistical power is limited.")

feature_cols = [c for c in df.columns if c != "composer"]

# ------------------------------------------------------------
# EXTRACT DISPUTED FEATURES
# ------------------------------------------------------------
disputed_feats = extract_features(DISPUTED_PATH)
if disputed_feats is None:
    raise ValueError("Could not extract features for disputed piece.")

row_d = pd.Series(disputed_feats)
print("\nExtracted disputed features.")


# ------------------------------------------------------------
# BOOTSTRAP p-value FUNCTION
# ------------------------------------------------------------
def bootstrap_pvalue(sample, observed, B=10000):
    """
    sample   : array of authentic Dussek values
    observed : value from the disputed piece

    Returns:
      p = fraction of bootstrap resamples whose
          sample statistic is as extreme or more extreme
          than the disputed statistic.
    """

    # The statistic we compare is absolute deviation from Dussek mean
    mu = np.mean(sample)
    observed_dev = abs(observed - mu)

    count = 0
    n = len(sample)

    for _ in range(B):
        resample = np.random.choice(sample, size=n, replace=True)
        mu_r = np.mean(resample)
        dev_r = abs(mu_r - mu)
        if dev_r >= observed_dev:
            count += 1

    return count / B


# ------------------------------------------------------------
# RUN BOOTSTRAP TEST FOR EVERY FEATURE
# ------------------------------------------------------------
results = []

for feat in feature_cols:
    sample = dus[feat].dropna().values
    if len(sample) < 4:
        continue

    disputed_val = row_d[feat]
    if pd.isna(disputed_val):
        continue

    p = bootstrap_pvalue(sample, disputed_val)

    results.append({
        "feature": feat,
        "dussek_mean": np.mean(sample),
        "disputed_value": disputed_val,
        "p_value": p
    })

res_df = pd.DataFrame(results).sort_values("p_value")

print("\n===========================================")
print("BOOTSTRAP p-values (lower = more outlier)")
print("===========================================\n")
print(res_df.to_string(index=False))

# ------------------------------------------------------------
# OPTIONAL: plot distribution for the most extreme feature
# ------------------------------------------------------------
worst_feat = res_df.iloc[0]["feature"]
sample = dus[worst_feat].dropna().values
disputed_val = row_d[worst_feat]

plt.figure(figsize=(8,5))
plt.hist(sample, bins=10, color="gray", alpha=0.7, label="Dussek authentic")
plt.axvline(disputed_val, color="red", linewidth=3, label="Disputed")
plt.title(f"Dussek distribution for '{worst_feat}'")
plt.xlabel("Feature value")
plt.ylabel("Count")
plt.legend()
plt.show()