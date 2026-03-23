"""
Cricket Match Winner Prediction Model
======================================
ML Pipeline Steps implemented in this file:
  Step 2: Data Preprocessing  → prepare_features() method
  Step 3: Split Data          → train_test_split() in train() method
  Step 4: Model Training      → model.fit() in train() method
  Step 5: Evaluate            → Accuracy, Cross-Validation in train() method
  Step 6: Predict             → predict() method
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os
import json


class MatchPredictor:
    """Predicts the winner of a cricket match."""

    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.accuracy = 0
        self.model_name = "Match Winner Predictor"

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  STEP 2: DATA PREPROCESSING                                    ║
    # ║  - Label Encoding: Convert text to numbers                     ║
    # ║  - Feature Engineering: Create strength_diff, toss features    ║
    # ║  - Feature Selection: Choose 12 most useful features           ║
    # ╚══════════════════════════════════════════════════════════════════╝
    def prepare_features(self, df):
        """Step 2: Data Preprocessing — encode, engineer, select features."""
        data = df.copy()

        # Step 2a: LABEL ENCODING — convert text to numbers
        categorical_cols = ["team1", "team2", "venue", "pitch_type",
                            "match_format", "toss_winner", "toss_decision"]

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[f"{col}_encoded"] = self.label_encoders[col].fit_transform(data[col])
            else:
                known_classes = set(self.label_encoders[col].classes_)
                data[col] = data[col].apply(lambda x: x if x in known_classes else self.label_encoders[col].classes_[0])
                data[f"{col}_encoded"] = self.label_encoders[col].transform(data[col])

        # Step 2b: FEATURE ENGINEERING — create new useful features
        data["strength_diff"] = data["team1_strength"] - data["team2_strength"]
        data["toss_winner_is_team1"] = (data["toss_winner"] == data["team1"]).astype(int)
        data["score_diff"] = data["team1_score"] - data["team2_score"]
        data["total_score"] = data["team1_score"] + data["team2_score"]

        # Step 2c: FEATURE SELECTION — choose 12 features
        self.feature_columns = [
            "team1_encoded", "team2_encoded", "venue_encoded",
            "pitch_type_encoded", "match_format_encoded",
            "toss_winner_is_team1", "toss_decision_encoded",
            "team1_home", "team2_home",
            "team1_strength", "team2_strength", "strength_diff",
        ]

        return data

    def train(self, df):
        """
        Steps 2-5 of the ML pipeline:
        Step 2: Preprocessing | Step 3: Split | Step 4: Train | Step 5: Evaluate
        """
        print("\n" + "=" * 60)
        print("🏏 MATCH WINNER PREDICTION MODEL")
        print("=" * 60)

        # ── STEP 2: Data Preprocessing ────────────────────────────────
        print("\n── Step 2: Data Preprocessing ──")
        data = self.prepare_features(df)
        self.label_encoders["winner"] = LabelEncoder()
        data["winner_encoded"] = self.label_encoders["winner"].fit_transform(data["winner"])
        print(f"   ✅ Encoded {len(self.label_encoders)} categorical columns")
        print(f"   ✅ Engineered: strength_diff, toss_winner_is_team1")
        print(f"   ✅ Selected {len(self.feature_columns)} features")

        # ── STEP 3: Split Data ────────────────────────────────────────
        print("\n── Step 3: Split Data ──")
        X = data[self.feature_columns]
        y = data["winner_encoded"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   Training: {len(X_train)} records (80%)")
        print(f"   Testing:  {len(X_test)} records (20%)")

        # ── STEP 4: Model Training ────────────────────────────────────
        print("\n── Step 4: Model Training ──")
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                min_samples_split=5, random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric="mlogloss", n_jobs=1
            ),
        }

        best_accuracy = 0
        best_model_name = ""

        # ── STEP 5: Evaluate ──────────────────────────────────────────
        print("\n── Step 5: Evaluate ──")
        for name, model in models.items():
            # Step 4: Train
            model.fit(X_train, y_train)
            # Step 5: Evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

            print(f"\n   🔹 {name}:")
            print(f"      Accuracy:    {acc:.4f} ({acc * 100:.1f}%)")
            print(f"      CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                self.model = model
                self.accuracy = acc

        print(f"\n   🏆 Best Model: {best_model_name} ({best_accuracy * 100:.1f}%)")

        if hasattr(self.model, "feature_importances_"):
            importances = pd.Series(
                self.model.feature_importances_, index=self.feature_columns
            ).sort_values(ascending=False)
            print(f"\n   📈 Top 5 Most Important Features:")
            for feat, imp in importances.head(5).items():
                print(f"      {feat}: {imp:.4f}")

        y_pred = self.model.predict(X_test)
        print(f"\n   📋 Classification Report:")
        unique_labels = sorted(set(y_test) | set(y_pred))
        target_names = [self.label_encoders["winner"].inverse_transform([l])[0] for l in unique_labels]
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

        return self.accuracy

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  STEP 6: PREDICT                                               ║
    # ║  User input → preprocess → model predicts → return winner      ║
    # ╚══════════════════════════════════════════════════════════════════╝
    def predict(self, team1, team2, venue, pitch_type, match_format,
                toss_winner, toss_decision, team1_home, team2_home,
                team1_strength, team2_strength):
        """Step 6: Predict — returns winner name and win probabilities."""
        # Step 6a: Create input DataFrame
        input_data = pd.DataFrame([{
            "team1": team1, "team2": team2, "venue": venue,
            "pitch_type": pitch_type, "match_format": match_format,
            "toss_winner": toss_winner, "toss_decision": toss_decision,
            "team1_home": team1_home, "team2_home": team2_home,
            "team1_strength": team1_strength, "team2_strength": team2_strength,
            "team1_score": 0, "team2_score": 0,
        }])

        # Step 6b: Preprocess (same encoding as training)
        data = self.prepare_features(input_data)
        X = data[self.feature_columns]

        # Step 6c: Model predicts winner class + probabilities
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Step 6d: Convert back to team name
        winner = self.label_encoders["winner"].inverse_transform([prediction])[0]
        classes = self.label_encoders["winner"].classes_
        prob_dict = {classes[i]: round(float(probabilities[i]) * 100, 1)
                     for i in range(len(classes))}
        relevant_probs = {
            team1: prob_dict.get(team1, 0),
            team2: prob_dict.get(team2, 0),
        }

        return winner, relevant_probs

    def save(self, path):
        """Save trained model to .pkl file."""
        joblib.dump({
            "model": self.model, "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns, "accuracy": self.accuracy,
        }, path)
        print(f"💾 Match model saved to {path}")

    def load(self, path):
        """Load trained model from .pkl file."""
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoders = data["label_encoders"]
        self.feature_columns = data["feature_columns"]
        self.accuracy = data["accuracy"]
        print(f"📂 Match model loaded from {path}")
