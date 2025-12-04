import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. Load Extracted Features
# ============================================================
df = pd.read_csv("./features/extracted_features.csv")

# Drop non-numeric column for covariance
feature_cols = [c for c in df.columns if c not in ["composer"]]
X = df[feature_cols]

print("Dataset shape:", df.shape)
print("Features:", feature_cols)


# ============================================================
# 2. Correlation Matrix (CS109-friendly heatmap)
# ============================================================
plt.figure(figsize=(14, 12))
sns.heatmap(X.corr(), vmin=-1, vmax=1, cmap="coolwarm", square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 3. Identify Feature Clusters
# (Simple grouping based on correlation strength)
# ============================================================
corr = X.corr().abs()

# Thresholds
strong_threshold = 0.6
moderate_threshold = 0.35

strong_pairs = []
moderate_pairs = []

for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        val = corr.iloc[i, j]
        if val >= strong_threshold:
            strong_pairs.append((feature_cols[i], feature_cols[j], round(val, 3)))
        elif val >= moderate_threshold:
            moderate_pairs.append((feature_cols[i], feature_cols[j], round(val, 3)))

print("\n===== STRONGLY CORRELATED FEATURE PAIRS (>|0.6|) =====")
for a, b, v in strong_pairs:
    print(f"{a:25s} <-> {b:25s}   corr = {v}")

print("\n===== MODERATELY CORRELATED FEATURE PAIRS (>|0.35|) =====")
for a, b, v in moderate_pairs:
    print(f"{a:25s} <-> {b:25s}   corr = {v}")


# ============================================================
# 4. Per-Composer Means (Feature Profiles)
# ============================================================
composer_means = df.groupby("composer")[feature_cols].mean()

plt.figure(figsize=(15, 8))
sns.heatmap(composer_means, annot=False, cmap="coolwarm")
plt.title("Composer Feature Profiles (Mean Values)")
plt.ylabel("Composer")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()


# ============================================================
# 5. Pairwise Scatter Plots for Key Feature Groups
# ============================================================

# ---- Pitch Cluster ----
pitch_features = ["mean_pitch", "std_pitch", "pitch_range"]
sns.pairplot(df, vars=pitch_features, hue="composer")
plt.suptitle("Pitch Feature Space — Pairwise Plots", y=1.02)
plt.show()

# ---- Chromaticity Cluster ----
chrom_features = ["accidental_ratio", "out_of_key_ratio", "accidental_change_rate"]
sns.pairplot(df, vars=chrom_features, hue="composer")
plt.suptitle("Chromaticity Feature Space — Pairwise Plots", y=1.02)
plt.show()

# ---- Texture Cluster ----
texture_features = ["chord_ratio", "harp_unfriendly_chord_ratio"]
sns.pairplot(df, vars=texture_features, hue="composer")
plt.suptitle("Texture/Chord Feature Space", y=1.02)
plt.show()

# ---- Melodic Interval Cluster ----
interval_features = ["mean_abs_interval", "step_ratio", "leap_ratio", "interval_entropy"]
sns.pairplot(df, vars=interval_features, hue="composer")
plt.suptitle("Melodic Interval Feature Space", y=1.02)
plt.show()