# CS109 Final Project — Composition Attribution via Probabilistic Modeling

This project uses probability, bootstrapping, and symbolic music analysis to determine whether a musical composition is stylistically consistent with its attributed composer.
The motivation comes from a real historical puzzle: Was the famous harp “Sonata in C Minor” actually written by Jan Ladislav Dussek — or by his wife, Sophia Corri Dussek, an accomplished harpist whose authorship was often erased?

Traditional musicology argues this piece “feels” idiomatic for harp writing.
This project asks: Can we quantify that?


## Motivation

Many female composers in the 18th–19th centuries published under their husbands’ names, or were denied authorship entirely. Because of this, modern datasets overrepresent already-famous male composers, while marginalized voices leave very small statistical footprints.

This project explores whether statistical evidence extracted from sheet music itself can shed light on misattributions — with the Dussek harp sonata as a case study.


## Project Overview

Step 1 — Extract symbolic musical features with music21

Instead of using audio, we analyze symbolic scores (MIDI / MusicXML), extracting ~20 quantitative features:
	•	Pitch features: mean pitch, range, variance
	•	Rhythmic features: duration statistics, IOI (inter-onset interval)
	•	Chromaticity: accidental ratio, out-of-key ratio, accidental-change rate
	•	Harmonic texture: chord ratio, harp-unfriendly chord ratio, span sizes
	•	Melodic interval statistics: mean interval, step ratio, leap ratio
	•	Entropy: interval entropy measuring predictability of melodic motion

These are the same kinds of features used in computational musicology, but implemented manually using music21 rather than sklearn.

⸻

Step 2 — Validate that features actually encode composer style

Before using the features on Dussek, we confirm they meaningfully separate known composers.

We processed 179 pieces across Bach, Mozart, Beethoven, Chopin, and Debussy, then:
- computed per-feature distributions
- discretized features into bins
- measured symmetric KL divergence between composer pairs

Example finding (interval entropy):
- Low KL (similar): Mozart ↔ Beethoven (~1.7)
- High KL (different): Bach ↔ Chopin (~3.9), Bach ↔ Debussy (~3.4)

This matches real music history: Classical composers cluster together; Romantic/Impressionist composers diverge. The extracted features genuinely encode composer identity → they’re valid for evaluating Dussek.

## Building a Composer Model for Dussek

Dussek has only 11 authenticated pieces, so we cannot assume Gaussianity or large-sample asymptotics.

Instead, we use a nonparametric approach:
- Naive Bayesian likelihood model across features

We fit a univariate distribution per feature using empirical binning.
- Bootstrapping for statistical significance

Using CS109’s bootstrap procedure, we estimate the probability that each feature value in the disputed sonata could have come from Dussek’s distribution.

This gives a per-feature p-value.

## Testing the Disputed Harp Sonata

We extract features from:
```
/data/instruments/harpists/dussek/sonata-en-do-menor-jan-ladislav-dussek.mid
```

High-evidence mismatches: wider range, larger leaps, more irregularity, and strongly harp-idiomatic chord shapes — all uncharacteristic of Dussek’s Classical keyboard writing.

Medium evidence (0.02–0.05)
- More chromatic movement
- More modulation

High p-values (≥ 0.20)
- IOI variation, chord density, accidental ratio
These features don’t carry stylistic weight → good sanity check.

Final Statistical Conclusion
- Posterior P(piece is by Dussek | features) ≈ 0
- The disputed sonata is a strong outlier relative to Dussek’s style
- The quantitative evidence supports the hypothesis that Sophia Corri Dussek may have been the true composer.

```
feature_extraction/          # music21 feature code
data/composers/              # MIDI files for Bach, Mozart, Beethoven, Chopin, Debussy, Dussek
data/instruments/            # Harp-specific datasets (incl. disputed sonata)
notebooks/                   # Jupyter notebooks: KL tests, plots, composer validation
models/                      # Likelihood model + bootstrap scripts
figures/                     # Generated plots for the report
README.md
```

## Running the Code
Run composer validation (KL divergence + visualizations)
```
python analyze_composers.py
```

Run Dussek attribution test (bootstrap + likelihood)
```
python test_dussek_attribution.py
```