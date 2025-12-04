import numpy as np
from music21 import converter, note, chord, key

"""
We need to create a dataset of scores that we can use to train our model that will classify the composer of a score.

To prepare our dataset for training, we need to extract the features from the scores. 
We will use the following features:
- Pitch intervals
- Pitch range
- Tempo
- Chromaticity
- Number of accidentals
- Key
- Time signature
- Number of measures
- Number of notes


Key giveaways that something is more for piano than harp:
1. lots of chromaticity/accidentals
2. chords that are too large to be played on harp (require 5 fingers in one hand)


We are using symbolic data, not audio data.

https://medium.com/data-science/midi-music-data-extraction-using-music21-and-word2vec-on-kaggle-cb383261cd4e

https://www.music21.org/music21docs/index.html 
"""
###############################################################################
# Utility: safely compute mean/std (0 if empty)
###############################################################################
def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0

def safe_std(x):
    return float(np.std(x)) if len(x) > 0 else 0.0


###############################################################################
# Feature Extraction
###############################################################################
def extract_features(filepath):
    """
    Extract global + local symbolic music features from a MIDI/MusicXML score.
    Returns a dictionary of scalar features.
    """

    try:
        score = converter.parse(filepath)
    except:
        print(f"Could not parse {filepath}")
        return None

    part = score.parts[0].flat
    events = [n for n in part.notesAndRests if isinstance(n, (note.Note, chord.Chord))]

    # -------------------------------------------------------------------------
    # Pitch features
    # -------------------------------------------------------------------------
    pitches = []
    for n in events:
        if isinstance(n, note.Note):
            pitches.append(n.pitch.midi)
        elif isinstance(n, chord.Chord):
            pitches.extend([p.midi for p in n.pitches])

    mean_pitch = safe_mean(pitches)
    std_pitch = safe_std(pitches)
    pitch_range = (max(pitches) - min(pitches)) if pitches else 0

    # -------------------------------------------------------------------------
    # Duration features
    # -------------------------------------------------------------------------
    durs = [float(n.quarterLength) for n in events]
    mean_duration = safe_mean(durs)
    std_duration = safe_std(durs)
    short_note_ratio = np.mean([d < 0.5 for d in durs]) if len(durs) else 0
    long_note_ratio  = np.mean([d >= 1.0 for d in durs]) if len(durs) else 0

    # -------------------------------------------------------------------------
    # Inter-onset intervals (IOI)
    # -------------------------------------------------------------------------
    notes_sorted = sorted(events, key=lambda n: float(n.offset))
    offsets = [float(n.offset) for n in notes_sorted]
    ioi = [offsets[i+1] - offsets[i] for i in range(len(offsets)-1)]

    mean_ioi = safe_mean(ioi)
    std_ioi  = safe_std(ioi)

    # -------------------------------------------------------------------------
    # Melodic notes (for interval and repetition analysis)
    # -------------------------------------------------------------------------
    melodic_notes = [n for n in notes_sorted if isinstance(n, note.Note)]

    # -------------------------------------------------------------------------
    # Chromaticity / Accidentals
    # -------------------------------------------------------------------------
    accidental_flags = []
    for n in events:
        if isinstance(n, note.Note):
            accidental_flags.append(n.pitch.accidental is not None)
        elif isinstance(n, chord.Chord):
            accidental_flags.extend([p.accidental is not None for p in n.pitches])

    accidental_ratio = np.mean(accidental_flags) if accidental_flags else 0

    # -------------------------------------------------------------------------
    # Out-of-key ratio (tonal deviation)
    # -------------------------------------------------------------------------
    try:
        k = score.analyze('key')
        scale_pitches = set(p.name for p in k.getPitches())
    except:
        scale_pitches = set()

    out_of_key_flags = []
    for n in events:
        if isinstance(n, note.Note):
            out_of_key_flags.append(n.pitch.name not in scale_pitches)
        elif isinstance(n, chord.Chord):
            out_of_key_flags.extend([p.name not in scale_pitches for p in n.pitches])

    out_of_key_ratio = np.mean(out_of_key_flags) if out_of_key_flags else 0

    # -------------------------------------------------------------------------
    # Chord texture + Harp-unfriendly chord spans (> 12 semitones)
    # -------------------------------------------------------------------------
    num_chords = 0
    large_chord_count = 0
    for n in events:
        if isinstance(n, chord.Chord):
            num_chords += 1
            span = max(p.midi for p in n.pitches) - min(p.midi for p in n.pitches)
            if span > 12:      # harp hand-span limit (approx)
                large_chord_count += 1

    chord_ratio = num_chords / len(events) if len(events) else 0
    harp_unfriendly_chord_ratio = (
        large_chord_count / num_chords if num_chords else 0
    )

    # -------------------------------------------------------------------------
    # Accidental-change rate (rapid pedal changes = harp-unfriendly)
    # -------------------------------------------------------------------------
    measure_acc_changes = []
    measures = score.parts[0].getElementsByClass('Measure')
    for m in measures:
        accs = []
        for n in m.notes:
            if isinstance(n, note.Note) and n.pitch.accidental:
                accs.append(n.pitch.accidental.name)
        # Count changes between successive accidentals
        changes = sum(accs[i] != accs[i+1] for i in range(len(accs)-1))
        measure_acc_changes.append(changes)

    accidental_change_rate = safe_mean(measure_acc_changes)

    # -------------------------------------------------------------------------
    # Rapid repeated-note ratio (harp cannot repeat same pitch very fast)
    # -------------------------------------------------------------------------
    rapid_repeats = 0
    total_pairs = max(0, len(melodic_notes) - 1)

    for i in range(total_pairs):
        same_pitch = melodic_notes[i].pitch.midi == melodic_notes[i+1].pitch.midi
        very_fast = (offsets[i+1] - offsets[i]) < 0.15  # threshold in quarterLength
        if same_pitch and very_fast:
            rapid_repeats += 1

    rapid_repetition_ratio = rapid_repeats / total_pairs if total_pairs else 0

    # -------------------------------------------------------------------------
    # Arpeggio marks (rolled chords → harp idiom)
    # -------------------------------------------------------------------------
    from music21.expressions import ArpeggioMark
    arpeggio_count = 0
    for n in events:
        if hasattr(n, "expressions"):
            if any(isinstance(exp, ArpeggioMark) for exp in n.expressions):
                arpeggio_count += 1

    arpeggio_mark_ratio = arpeggio_count / len(events) if len(events) else 0

    # -------------------------------------------------------------------------
    # Local melodic intervals + interval entropy
    # -------------------------------------------------------------------------
    intervals = []
    for i in range(1, len(melodic_notes)):
        p1 = melodic_notes[i-1].pitch.midi
        p2 = melodic_notes[i].pitch.midi
        intervals.append(p2 - p1)

    abs_intervals = [abs(iv) for iv in intervals]
    mean_abs_interval = safe_mean(abs_intervals)
    step_ratio = np.mean([a <= 2 for a in abs_intervals]) if abs_intervals else 0
    leap_ratio = np.mean([a >= 5 for a in abs_intervals]) if abs_intervals else 0

    if len(intervals) > 0:
        unique, counts = np.unique(intervals, return_counts=True)
        probs = counts / counts.sum()
        interval_entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        interval_entropy = 0.0

    # -------------------------------------------------------------------------
    # Final packaged features (with concise explanations)
    # -------------------------------------------------------------------------
    features = {
        "mean_pitch": mean_pitch,                         # avg pitch height; instrument tessitura
        "std_pitch": std_pitch,                           # pitch variability; stylistic marker
        "pitch_range": pitch_range,                       # max-min pitch; harp vs piano range

        "mean_duration": mean_duration,                   # typical note length; rhythmic density
        "std_duration": std_duration,                     # rhythmic variability
        "short_note_ratio": short_note_ratio,             # % short notes; dense textures
        "long_note_ratio": long_note_ratio,               # % long notes; sustained phrasing

        "mean_ioi": mean_ioi,                             # mean spacing between notes
        "std_ioi": std_ioi,                               # rhythmic variability

        "accidental_ratio": accidental_ratio,             # overall chromaticity; harp-unfriendly
        "out_of_key_ratio": out_of_key_ratio,             # tonal deviation; increases difficulty

        "chord_ratio": chord_ratio,                       # % chord events; piano texture
        "harp_unfriendly_chord_ratio": harp_unfriendly_chord_ratio,  
                                                          # chords >12 semitones → not playable on harp

        "accidental_change_rate": accidental_change_rate, # rapid accidental swings → rapid pedal changes
        "rapid_repetition_ratio": rapid_repetition_ratio, # same pitch repeated VERY fast → hard for harp
        "arpeggio_mark_ratio": arpeggio_mark_ratio,       # rolled chords symbol → harp idiom indicator

        "mean_abs_interval": mean_abs_interval,           # avg melodic step size
        "step_ratio": step_ratio,                         # % intervals ≤2 semitones; smoother motion
        "leap_ratio": leap_ratio,                         # % intervals ≥5 semitones; pianistic leaps
        "interval_entropy": interval_entropy              # unpredictability of melodic intervals
    }

    return features