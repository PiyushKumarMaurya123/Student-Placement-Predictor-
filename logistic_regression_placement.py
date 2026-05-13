# =============================================================================
# LOGISTIC REGRESSION — Student Placement Prediction
# Dataset: student_placement_salary_elite_v2.csv (9000 rows, 20 columns)
# Target : 'placed' (1 = placed, 0 = not placed)
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# -----------------------------------------------------------------------------
# pandas  → load and manipulate tabular data
# numpy   → numerical operations
# sklearn → machine learning tools (preprocessing, model, evaluation)
# matplotlib/seaborn → visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

print("=" * 60)
print("  LOGISTIC REGRESSION — Student Placement Prediction")
print("=" * 60)

# -----------------------------------------------------------------------------
# STEP 2: LOAD DATA
# -----------------------------------------------------------------------------
# We load the CSV into a DataFrame (like a spreadsheet in Python).
# Each row = one student, each column = one feature.

df = pd.read_csv("student_placement_salary_elite_v2.csv")

print(f"\n📦 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nColumn names:\n{df.columns.tolist()}")

# -----------------------------------------------------------------------------
# STEP 3: UNDERSTAND THE TARGET VARIABLE
# -----------------------------------------------------------------------------
# 'placed' is our target (what we want to predict):
#   1 → Student was placed (got a job)
#   0 → Student was NOT placed
#
# This is a BINARY CLASSIFICATION problem — exactly what Logistic Regression
# is designed for.

print("\n\n📊 Target Variable (placed):")
print(df['placed'].value_counts())
print(f"\nPlacement Rate: {df['placed'].mean()*100:.1f}%")

# -----------------------------------------------------------------------------
# STEP 4: FEATURE SELECTION
# -----------------------------------------------------------------------------
# We choose which columns to use as INPUT features for the model.
#
# We DROP:
#   - 'student_id'   → just an ID, no predictive value
#   - 'company_type' → only available AFTER placement (data leakage!)
#   - 'job_role'     → same reason — only known after placement
#   - 'salary_lpa'   → 0 for unplaced students, leaks the answer
#   - 'placed'       → this is our TARGET, not an input
#
# ⚠️ Data Leakage: using features that are only known AFTER the event
#    you're predicting causes the model to "cheat" and gives fake accuracy.

DROP_COLS = ['student_id', 'company_type', 'job_role', 'salary_lpa', 'placed']

features   = [col for col in df.columns if col not in DROP_COLS]
TARGET_COL = 'placed'

print(f"\n\n✅ Features used ({len(features)}):\n{features}")

# -----------------------------------------------------------------------------
# STEP 5: ENCODE CATEGORICAL VARIABLES
# -----------------------------------------------------------------------------
# Machine learning models only understand NUMBERS.
# 'branch' column has text values like "CSE", "IT", "Civil" etc.
# We use LabelEncoder to convert them to integers: CSE=0, IT=1, Civil=2 ...
#
# For a more advanced approach you'd use One-Hot Encoding (pd.get_dummies),
# but LabelEncoder is simpler for a single column with many categories.

le = LabelEncoder()
df['branch'] = le.fit_transform(df['branch'])   # e.g. Civil→0, CSE→1, IT→2 ...

print(f"\n\n🔠 Branch encoded. Unique values: {df['branch'].unique()}")

# -----------------------------------------------------------------------------
# STEP 6: PREPARE X (features) and y (target)
# -----------------------------------------------------------------------------
# X → the input matrix  (9000 rows × 15 features)
# y → the output vector (9000 values: 0 or 1)

X = df[features]   # All input features
y = df[TARGET_COL] # What we want to predict

print(f"\n\n📐 X shape: {X.shape}")
print(f"📐 y shape: {y.shape}")
print(f"\nSample X (first row):\n{X.iloc[0].to_dict()}")

# -----------------------------------------------------------------------------
# STEP 7: SPLIT INTO TRAIN AND TEST SETS
# -----------------------------------------------------------------------------
# We split data into:
#   Training set (80%) → model LEARNS from this
#   Test set     (20%) → model is EVALUATED on this (never seen before)
#
# Why split? To check if our model generalizes to new data.
# If we trained and tested on the same data, the model would just memorize!
#
# stratify=y → ensures both train & test have the same % of placed/unplaced
# random_state=42 → seeds the random split so results are reproducible

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42,
    stratify=y        # maintain class balance
)

print(f"\n\n✂️  Train/Test Split:")
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")
print(f"   Train placement rate: {y_train.mean()*100:.1f}%")
print(f"   Test  placement rate: {y_test.mean()*100:.1f}%")

# -----------------------------------------------------------------------------
# STEP 8: FEATURE SCALING (Standardization)
# -----------------------------------------------------------------------------
# Logistic Regression is sensitive to the SCALE of features.
# Example:
#   - 'cgpa' ranges from 5 to 10  (small scale)
#   - 'coding_score' ranges from 0 to 100  (large scale)
#
# Without scaling, 'coding_score' would dominate just because it's bigger.
# StandardScaler transforms each feature to: mean=0, std=1
#
# Formula: z = (x - mean) / std
#
# IMPORTANT: fit only on TRAINING data, then transform BOTH train and test.
# Fitting on test data would leak test information into training.

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)  # learn mean/std from train, then scale
X_test  = scaler.transform(X_test)       # use SAME mean/std to scale test

print(f"\n\n📏 Feature Scaling applied (StandardScaler)")
print(f"   Each feature now has mean≈0 and std≈1")

# -----------------------------------------------------------------------------
# STEP 9: TRAIN LOGISTIC REGRESSION MODEL
# -----------------------------------------------------------------------------
# LogisticRegression learns a mathematical formula:
#
#   P(placed=1) = sigmoid(w0 + w1*cgpa + w2*coding_score + ... )
#
# Where sigmoid(z) = 1 / (1 + e^(-z))  → squishes any number into [0, 1]
#
# During training, it finds the best weights (w0, w1, w2, ...) that
# maximize the probability of correct predictions.
#
# Parameters:
#   C=1.0         → regularization strength (prevents overfitting)
#   max_iter=1000 → max iterations for the optimizer to converge
#   random_state  → for reproducibility

model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print(f"\n\n🤖 Model trained successfully!")
print(f"   Number of features : {model.n_features_in_}")
print(f"   Number of iterations: {model.n_iter_[0]}")

# -----------------------------------------------------------------------------
# STEP 10: MAKE PREDICTIONS
# -----------------------------------------------------------------------------
# model.predict()       → gives class label (0 or 1)
# model.predict_proba() → gives probability for each class [P(0), P(1)]
#
# We use a threshold of 0.5:
#   if P(placed=1) >= 0.5 → predict 1 (placed)
#   if P(placed=1) <  0.5 → predict 0 (not placed)

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of being placed

print(f"\n\n🔮 Sample Predictions (first 10):")
for i in range(10):
    actual    = y_test.iloc[i]
    predicted = y_pred[i]
    prob      = y_pred_proba[i]
    status    = "✅" if actual == predicted else "❌"
    print(f"   {status} Actual: {actual}  Predicted: {predicted}  Probability: {prob:.2f}")

# -----------------------------------------------------------------------------
# STEP 11: EVALUATE THE MODEL
# -----------------------------------------------------------------------------
# Several metrics to understand model performance:

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_proba)

print(f"\n\n📈 MODEL EVALUATION")
print(f"{'='*40}")
print(f"  Accuracy : {accuracy*100:.2f}%")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

# WHAT THESE METRICS MEAN:
# ─────────────────────────────────────────────────────────────────────────────
# Accuracy   → % of total correct predictions
#              Problem: misleading if classes are imbalanced!
#
# Precision  → Of students predicted as "placed", how many actually were?
#              High precision = fewer false alarms
#
# Recall     → Of students actually placed, how many did we catch?
#              High recall = fewer missed placements
#
# F1-Score   → Harmonic mean of Precision and Recall (balance of both)
#
# ROC-AUC    → Measures how well the model separates the two classes.
#              1.0 = perfect, 0.5 = random guessing
# ─────────────────────────────────────────────────────────────────────────────

# -----------------------------------------------------------------------------
# STEP 12: VISUALIZATIONS
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Logistic Regression — Student Placement Prediction", fontsize=14, fontweight='bold')

# --- Plot 1: Confusion Matrix ---
# Shows counts of TP, TN, FP, FN predictions
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Placed", "Placed"])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix\n(How many did we get right/wrong?)")

# --- Plot 2: ROC Curve ---
# Shows the tradeoff between True Positive Rate and False Positive Rate
# The more the curve bows toward top-left, the better the model
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.3f}')
axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random (AUC=0.5)')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve\n(Model vs Random Guessing)')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

# --- Plot 3: Feature Importances (Coefficients) ---
# In Logistic Regression, coefficients tell us each feature's impact.
# Positive coeff → increases P(placed)
# Negative coeff → decreases P(placed)
feature_names = features
coefficients  = model.coef_[0]
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)

colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
axes[2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
axes[2].axvline(x=0, color='black', linewidth=0.8)
axes[2].set_xlabel('Coefficient Value')
axes[2].set_title('Feature Coefficients\n(Green=Helps, Red=Hurts Placement)')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("logistic_regression_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n\n📊 Plots saved to logistic_regression_results.png")

# -----------------------------------------------------------------------------
# STEP 13: PREDICT FOR A NEW STUDENT
# -----------------------------------------------------------------------------
# Let's simulate predicting placement for a brand-new student.

print("\n\n🎓 PREDICTING FOR A NEW STUDENT:")
print("-" * 40)

# branch 1 = CSE (after encoding)
new_student = pd.DataFrame([{
    'cgpa': 8.5,
    'branch': 1,            # CSE
    'college_tier': 1,
    'python_skill': 4,
    'dsa_skill': 4,
    'ml_skill': 3,
    'web_dev_skill': 3,
    'coding_score': 85.0,
    'communication_score': 80.0,
    'aptitude_score': 78.0,
    'internships': 2,
    'projects': 3,
    'backlogs': 0,
    'resume_score': 88.0,
    'skill_score': 4
}])

# Scale using the SAME scaler fitted on training data
new_student_scaled = scaler.transform(new_student)

pred_label = model.predict(new_student_scaled)[0]
pred_proba = model.predict_proba(new_student_scaled)[0][1]

print(f"  Placement Probability : {pred_proba*100:.1f}%")
print(f"  Prediction            : {'✅ PLACED' if pred_label == 1 else '❌ NOT PLACED'}")

print("\n\n✅ Done! Check logistic_regression_results.png for visualizations.")
