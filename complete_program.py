import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================
# 1. LOAD FEATURE DATA
# =============================================================
df = pd.read_csv("./features/extracted_features.csv")

X = df.drop(columns=["composer"])
y = df["composer"]

feature_names = X.columns


# =============================================================
# 2. STANDARDIZE FEATURES
# =============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =============================================================
# 3. COVARIANCE MATRIX (CS109 STYLE)
# =============================================================
cov_matrix = np.cov(X_scaled, rowvar=False)

plt.figure(figsize=(10,8))
sns.heatmap(cov_matrix, 
            xticklabels=feature_names, 
            yticklabels=feature_names, 
            cmap="coolwarm", 
            center=0)
plt.title("Feature Covariance Matrix")
plt.show()


# =============================================================
# 4. PCA VISUALIZATION
# =============================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="Set2")
plt.title("PCA of Musical Feature Space")
plt.xlabel("PC1 ({:.2f}% var)".format(pca.explained_variance_ratio_[0]*100))
plt.ylabel("PC2 ({:.2f}% var)".format(pca.explained_variance_ratio_[1]*100))
plt.legend()
plt.show()


# =============================================================
# 5. TRAIN / TEST SPLIT
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, stratify=y, random_state=42
)


# =============================================================
# 6. DEFINE MODELS
# =============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, multi_class="multinomial"),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300)
}


# =============================================================
# 7. TRAIN + EVALUATE ALL MODELS (CS109 LECTURE 22 STYLE)
# =============================================================
for name, model in models.items():
    print("\n==============================")
    print(f"MODEL: {name}")
    print("==============================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# =============================================================
# 8. CALIBRATION PLOT (Lecture 22: "Comparing Classifiers: Calibration")
# =============================================================
lr = models["Logistic Regression"]
probs = lr.predict_proba(X_test)

# plot calibration for each composer (prob of correct class)
plt.figure(figsize=(8,6))
for i, comp in enumerate(lr.classes_):
    y_true_binary = (y_test == comp).astype(int)
    prob_true, prob_pred = calibration_curve(y_true_binary, probs[:,i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=comp)

plt.plot([0,1],[0,1],'k--')
plt.xlabel("Predicted Probability")
plt.ylabel("Actual Frequency (Percent Y=1)")
plt.title("Calibration Curves — Logistic Regression")
plt.legend()
plt.show()