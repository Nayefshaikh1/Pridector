"""
Cricket Match Winner Prediction Model
======================================
Uses Random Forest and XGBoost to predict match outcomes.
Features: team strengths, venue, pitch type, toss, home advantage, etc.
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

    def prepare_features(self, df):
        """Prepare features for training/prediction."""
        data = df.copy()

        # Encode categorical variables
        categorical_cols = ["team1", "team2", "venue", "pitch_type",
                            "match_format", "toss_winner", "toss_decision"]

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[f"{col}_encoded"] = self.label_encoders[col].fit_transform(data[col])
            else:
                # Handle unseen labels
                known_classes = set(self.label_encoders[col].classes_)
                data[col] = data[col].apply(lambda x: x if x in known_classes else self.label_encoders[col].classes_[0])
                data[f"{col}_encoded"] = self.label_encoders[col].transform(data[col])

        # Feature engineering
        data["strength_diff"] = data["team1_strength"] - data["team2_strength"]
        data["toss_winner_is_team1"] = (data["toss_winner"] == data["team1"]).astype(int)
        data["score_diff"] = data["team1_score"] - data["team2_score"]
        data["total_score"] = data["team1_score"] + data["team2_score"]

        self.feature_columns = [
            "team1_encoded", "team2_encoded", "venue_encoded",
            "pitch_type_encoded", "match_format_encoded",
            "toss_winner_is_team1", "toss_decision_encoded",
            "team1_home", "team2_home",
            "team1_strength", "team2_strength", "strength_diff",
        ]

        return data

    def train(self, df):
        """Train the match prediction model."""
        print("\n" + "=" * 60)
        print("🏏 TRAINING MATCH WINNER PREDICTION MODEL")
        print("=" * 60)

        data = self.prepare_features(df)

        # Encode target
        self.label_encoders["winner"] = LabelEncoder()
        data["winner_encoded"] = self.label_encoders["winner"].fit_transform(data["winner"])

        X = data[self.feature_columns]
        y = data["winner_encoded"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n📊 Dataset: {len(X_train)} training, {len(X_test)} testing samples")

        # Train multiple models and pick the best
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
                random_state=42, eval_metric="mlogloss",
                n_jobs=1
            ),
        }

        best_accuracy = 0
        best_model_name = ""

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

            print(f"\n🔹 {name}:")
            print(f"   Test Accuracy:  {acc:.4f} ({acc * 100:.1f}%)")
            print(f"   CV Accuracy:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                self.model = model
                self.accuracy = acc

        print(f"\n🏆 Best Model: {best_model_name} ({best_accuracy * 100:.1f}%)")

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            importances = pd.Series(
                self.model.feature_importances_, index=self.feature_columns
            ).sort_values(ascending=False)
            print(f"\n📈 Top Feature Importances:")
            for feat, imp in importances.head(5).items():
                print(f"   {feat}: {imp:.4f}")

        # Detailed report
        y_pred = self.model.predict(X_test)
        print(f"\n📋 Classification Report:")
        # Get unique labels that appear in test set
        unique_labels = sorted(set(y_test) | set(y_pred))
        target_names = [self.label_encoders["winner"].inverse_transform([l])[0] for l in unique_labels]
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

        return self.accuracy

    def predict(self, team1, team2, venue, pitch_type, match_format,
                toss_winner, toss_decision, team1_home, team2_home,
                team1_strength, team2_strength):
        """Predict match winner."""
        input_data = pd.DataFrame([{
            "team1": team1, "team2": team2, "venue": venue,
            "pitch_type": pitch_type, "match_format": match_format,
            "toss_winner": toss_winner, "toss_decision": toss_decision,
            "team1_home": team1_home, "team2_home": team2_home,
            "team1_strength": team1_strength, "team2_strength": team2_strength,
            "team1_score": 0, "team2_score": 0,  # Not used for prediction features we selected
        }])

        data = self.prepare_features(input_data)
        X = data[self.feature_columns]

        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        winner = self.label_encoders["winner"].inverse_transform([prediction])[0]
        classes = self.label_encoders["winner"].classes_

        prob_dict = {classes[i]: round(float(probabilities[i]) * 100, 1)
                     for i in range(len(classes))}

        # Get only relevant probabilities
        relevant_probs = {
            team1: prob_dict.get(team1, 0),
            team2: prob_dict.get(team2, 0),
        }

        return winner, relevant_probs

    def save(self, path):
        """Save the model and encoders."""
        joblib.dump({
            "model": self.model,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "accuracy": self.accuracy,
        }, path)
        print(f"💾 Match model saved to {path}")

    def load(self, path):
        """Load the model and encoders."""
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoders = data["label_encoders"]
        self.feature_columns = data["feature_columns"]
        self.accuracy = data["accuracy"]
        print(f"📂 Match model loaded from {path}")
