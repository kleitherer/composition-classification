import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

df = pd.read_csv("./features/extracted_features.csv")
sns.set(style="whitegrid", font_scale=1.2)

def midi_to_note(midi):
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi // 12) - 1
    return f"{names[midi % 12]}{octave}"

NOTE_60 = midi_to_note(60)


# ============================================================================
# 1. MEAN PITCH PLOT (SIDE NOTES)
# ============================================================================
fig = plt.figure(figsize=(14, 7))
gs = GridSpec(1, 2, width_ratios=[3, 1])

ax = fig.add_subplot(gs[0])
sns.kdeplot(data=df, x="mean_pitch", hue="composer", fill=True, alpha=0.3, ax=ax)

ax.set_title(f"Mean Pitch Distribution (MIDI 60 = {NOTE_60})")
ax.set_xlabel("Mean Pitch (MIDI)")
ax.set_ylabel("Density")

ax.axvline(60, color="black", linestyle="--", alpha=0.6)

# ---- SIDE PANEL ----
side = fig.add_subplot(gs[1])
side.axis("off")
side.text(0, 0.95, "Interpretation", fontsize=15, fontweight="bold")

side.text(
    0, 0.75,
    "• Values around 60 = middle register\n"
    "• Higher mean pitch → upper register\n"
    "• Lower mean pitch → bass-heavy writing\n"
    "• Useful for distinguishing:\n"
    "    - Bach + Mozart (balanced range)\n"
    "    - Debussy (often more extreme ranges)",
    fontsize=11,
    va="top"
)

plt.tight_layout()
plt.show()



# ============================================================================
# 2. CHROMATICITY: ACCIDENTALS vs OUT-OF-KEY
# ============================================================================
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(1, 2, width_ratios=[3, 1])

ax = fig.add_subplot(gs[0])
sns.scatterplot(
    data=df, x="accidental_ratio", y="out_of_key_ratio",
    hue="composer", alpha=0.85, s=80, ax=ax
)

ax.set_title("Chromaticity vs Harmonic Stability")
ax.set_xlabel("Accidental Ratio\n(local chromatic color)")
ax.set_ylabel("Out-of-Key Ratio\n(harmonic instability)")

ax.axvline(df.accidental_ratio.mean(), linestyle="--", color="gray", alpha=0.5)
ax.axhline(df.out_of_key_ratio.mean(), linestyle="--", color="gray", alpha=0.5)

# ---- SIDE PANEL ----
side = fig.add_subplot(gs[1])
side.axis("off")
side.text(0, 0.95, "Interpretation", fontsize=15, fontweight="bold")

side.text(
    0, 0.78,
    "Accidental Ratio:\n"
    "• Sharps/flats on individual notes.\n"
    "• High = chromatic color (decorative or expressive).",
    fontsize=11
)

side.text(
    0, 0.55,
    "Out-of-Key Ratio:\n"
    "• Notes not belonging to the home scale.\n"
    "• High = modulation, harmonic shifts.",
    fontsize=11
)

side.text(
    0, 0.30,
    "Quadrants:\n"
    "• Low/Low → Diatonic stability (Mozart).\n"
    "• High Acc / Low OOK → Ornamentation (Bach).\n"
    "• High/High → Chromatic harmony (Debussy & Chopin).",
    fontsize=11
)

plt.tight_layout()
plt.show()



# ============================================================================
# 3. CHORD RATIO vs HARP UNFRIENDLY CHORD RATIO
# ============================================================================
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(1, 2, width_ratios=[3, 1])

ax = fig.add_subplot(gs[0])
sns.scatterplot(
    data=df, x="chord_ratio", y="harp_unfriendly_chord_ratio",
    hue="composer", alpha=0.85, s=80, ax=ax
)

ax.set_title("Chord Density vs Harp-Unfriendly Chords")
ax.set_xlabel("Chord Ratio")
ax.set_ylabel("Harp-Unfriendly Chord Ratio\n(chord spans > 10th)")

# ---- SIDE PANEL ----
side = fig.add_subplot(gs[1])
side.axis("off")
side.text(0, 0.95, "Interpretation", fontsize=15, fontweight="bold")

side.text(
    0, 0.78,
    "Chord Ratio:\n"
    "• How many events contain simultaneous notes.\n"
    "• High = dense texture (pianistic).",
    fontsize=11
)

side.text(
    0, 0.50,
    "Harp-Unfriendly Ratio:\n"
    "• Chords spanning beyond 10th\n"
    "• Or shapes physically impossible for one hand.",
    fontsize=11
)

side.text(
    0, 0.25,
    "Regions:\n"
    "• Low/Low → Harp-friendly.\n"
    "• High/Low → Pianistic but playable.\n"
    "• High/High → Harp difficulty spike.\n"
    "• Debussy often pushes towards upper region.",
    fontsize=11
)

plt.tight_layout()
plt.show()



# ============================================================================
# 4. INTERVAL SIZE vs INTERVAL ENTROPY (EXPANDED)
# ============================================================================
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(1, 2, width_ratios=[3, 1])

ax = fig.add_subplot(gs[0])
sns.scatterplot(
    data=df, x="mean_abs_interval", y="interval_entropy",
    hue="composer", alpha=0.85, s=80, ax=ax
)

ax.set_title("Melodic Interval Size vs Interval Entropy")
ax.set_xlabel("Mean Absolute Interval (semitones)")
ax.set_ylabel("Interval Entropy (bits)")

# ---- SIDE PANEL ----
side = fig.add_subplot(gs[1])
side.axis("off")
side.text(0, 0.95, "Interpretation", fontsize=15, fontweight="bold")

side.text(
    0, 0.75,
    "Mean Absolute Interval:\n"
    "• Average step size between consecutive notes.\n"
    "• Low = stepwise (Mozart, Bach).\n"
    "• High = frequent leaps (Beethoven, Debussy).",
    fontsize=11
)

side.text(
    0, 0.45,
    "Interval Entropy:\n"
    "• Statistical unpredictability of interval choices.\n"
    "• Low entropy → predictable interval patterns\n"
    "    (e.g. sequences, scales, smooth melodies).\n"
    "• High entropy → irregular leaps, unstable contour\n"
    "    (e.g. Debussy, late Beethoven).",
    fontsize=11
)

side.text(
    0, 0.15,
    "Joint Interpretation:\n"
    "• Low interval + low entropy → Classical clarity.\n"
    "• High interval + high entropy → Romantic/Impressionist freedom.\n"
    "• Chopin often: medium leaps + high entropy\n"
    "    → expressive but unpredictable lines.",
    fontsize=11
)

plt.tight_layout()
plt.show()