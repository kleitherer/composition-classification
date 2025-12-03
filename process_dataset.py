import os
import pandas as pd
from feature_extraction import extract_features

root_dir = "./data/composers/"
rows = []

for composer in os.listdir(root_dir):
    composer_path = os.path.join(root_dir, composer)
    for filename in os.listdir(composer_path):
        if filename.endswith(".mid") or filename.endswith(".midi") or filename.endswith(".xml"):
            fp = os.path.join(composer_path, filename)
            feats = extract_features(fp)
            if feats is not None:
                feats["composer"] = composer
                rows.append(feats)

df = pd.DataFrame(rows)
df.to_csv("./features/extracted_features.csv", index=False)
print("Saved features for", len(df), "pieces.")