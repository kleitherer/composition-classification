import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from feature_extraction import extract_features

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
FEATURE_CSV = "./features/extracted_features.csv"
DISPUTED_PATH = "/Users/kaitlynleitherer/Desktop/CS109/final_project/data/instruments/harpists/dussek/sonata-en-do-menor-jan-ladislav-dussek.mid"


# ============================================================
# 1. Load all composer features
# ============================================================
df = pd.read_csv(FEATURE_CSV)

# ensure Dussek exists
if "dussek" not in df["composer"].unique():
    raise ValueError("Dussek not found in extracted_features.csv")

df_dussek = df[df["composer"] == "dussek"]
df_others = df[df["composer"] != "dussek"]

print("Loaded:", len(df_dussek), "authenticated Dussek pieces")


# ============================================================
# 2. Extract disputed piece features + save them separately
# ============================================================
disputed_feats = extract_features(DISPUTED_PATH)
disputed_df = pd.DataFrame([disputed_feats])
disputed_df.to_csv("./features/disputed_features.csv", index=False)

print("\nDisputed piece features saved → ./features/disputed_features.csv\n")
print(disputed_df.T)


# ============================================================
# 3. Compute Dussek feature means (for visual comparison)
# ============================================================
feature_cols = [c for c in df.columns if c not in ["composer"]]

dussek_means = df_dussek[feature_cols].mean()
dussek_stds  = df_dussek[feature_cols].std()

# print nicely
print("\n=== Authentic Dussek Feature Means ===\n")
print(dussek_means)


# ============================================================
# 4. VISUALIZATION: Composer Comparison for Key Features
# ============================================================

# Choose musically meaningful features
KEY_FEATURES = [
    "pitch_range",
    "mean_duration",
    "std_duration",
    "mean_abs_interval",
    "interval_entropy",
    "harp_unfriendly_chord_ratio",
    "accidental_change_rate",
    "out_of_key_ratio"
]

# Subset for clean visualization (Dussek vs. others)
viz_df = df.copy()
viz_df["composer_simple"] = viz_df["composer"].apply(
    lambda x: "dussek" if x == "dussek" else "other composers"
)


# ------------------------------------------------------------
# Create comparison plots
# ------------------------------------------------------------
os.makedirs("./figures/dussek_profile/", exist_ok=True)

for feat in KEY_FEATURES:

    plt.figure(figsize=(9,5))
    sns.kdeplot(
        data=viz_df,
        x=feat,
        hue="composer_simple",
        fill=True,
        alpha=0.4,
        linewidth=1.5,
        palette={"dussek":"darkred", "other composers":"gray"}
    )

    # mark Dussek mean
    plt.axvline(dussek_means[feat], color="darkred", linestyle="--", linewidth=2,
                label="Dussek mean")

    # mark disputed piece
    plt.axvline(disputed_df[feat].iloc[0], color="blue", linestyle="-", linewidth=2,
                label="Disputed sonata")

    plt.title(f"{feat} — Dussek vs. Other Composers")
    plt.xlabel(feat)
    plt.ylabel("density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"./figures/dussek_profile/{feat}.png", dpi=300)
    plt.close()

print("\nSaved comparison plots → ./figures/dussek_profile/")
print("Features visualized:", KEY_FEATURES)