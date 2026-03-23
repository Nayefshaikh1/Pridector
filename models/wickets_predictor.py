"""
Cricket Player Wickets Prediction Model
========================================
ML Pipeline Steps implemented in this file:
  Step 2: Data Preprocessing  → prepare_features() method
  Step 3: Split Data          → train_test_split() in train() method
  Step 4: Model Training      → model.fit() in train() method
  Step 5: Evaluate            → MAE, RMSE, R², Cross-Validation in train() method
  Step 6: Predict             → predict() method
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


class WicketsPredictor:
    """Predicts wickets taken by a bowler in a match."""

    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.mae = 0
        self.r2 = 0
        self.model_name = "Player Wickets Predictor"

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  STEP 2: DATA PREPROCESSING                                    ║
    # ║  - Label Encoding: Convert bowler names, teams to numbers      ║
    # ║  - Feature Engineering: avg_econ_ratio, pitch_favorable        ║
    # ║  - Feature Selection: Choose 16 features for prediction        ║
    # ╚══════════════════════════════════════════════════════════════════╝
    def prepare_features(self, df):
        """Step 2: Data Preprocessing — encode, engineer, select features."""
        data = df.copy()

        # Step 2a: LABEL ENCODING
        categorical_cols = ["bowler", "team", "opponent", "venue",
                            "pitch_type", "match_format", "bowling_type"]

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[f"{col}_encoded"] = self.label_encoders[col].fit_transform(data[col])
            else:
                known_classes = set(self.label_encoders[col].classes_)
                data[col] = data[col].apply(lambda x: x if x in known_classes else self.label_encoders[col].classes_[0])
                data[f"{col}_encoded"] = self.label_encoders[col].transform(data[col])

        # Step 2b: FEATURE ENGINEERING
        # Ratio features — help model understand bowler effectiveness
        data["avg_econ_ratio"] = data["bowling_avg"] / (data["bowling_econ"] + 0.1)
        data["sr_econ_ratio"] = data["bowling_sr"] / (data["bowling_econ"] + 0.1)

        # Pitch favorability — does the pitch suit this bowler's type?
        # Fast bowler on pace pitch = favorable, Spin bowler on spin pitch = favorable
        data["pitch_favorable"] = 0
        data.loc[(data["pitch_type"] == "pace") & (data["bowling_type"] == "fast"), "pitch_favorable"] = 1
        data.loc[(data["pitch_type"] == "spin") & (data["bowling_type"] == "spin"), "pitch_favorable"] = 1

        # Step 2c: FEATURE SELECTION — 16 features
        self.feature_columns = [
            "bowler_encoded", "team_encoded", "opponent_encoded",
            "venue_encoded", "pitch_type_encoded", "match_format_encoded",
            "bowling_avg", "bowling_sr", "bowling_econ",
            "bowling_type_encoded", "is_home", "opponent_strength",
            "overs_bowled", "avg_econ_ratio", "sr_econ_ratio",
            "pitch_favorable",
        ]

        return data

    def train(self, df):
        """
        Steps 2-5 of the ML pipeline:
        Step 2: Preprocessing | Step 3: Split | Step 4: Train | Step 5: Evaluate
        """
        print("\n" + "=" * 60)
        print("🏏 PLAYER WICKETS PREDICTION MODEL")
        print("=" * 60)

        # ── STEP 2: Data Preprocessing ────────────────────────────────
        print("\n── Step 2: Data Preprocessing ──")
        data = self.prepare_features(df)
        print(f"   ✅ Encoded {len(self.label_encoders)} categorical columns")
        print(f"   ✅ Engineered: avg_econ_ratio, sr_econ_ratio, pitch_favorable")
        print(f"   ✅ Selected {len(self.feature_columns)} features")

        # ── STEP 3: Split Data ────────────────────────────────────────
        print("\n── Step 3: Split Data ──")
        X = data[self.feature_columns]
        y = data["wickets_taken"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   Training: {len(X_train)} records (80%)")
        print(f"   Testing:  {len(X_test)} records (20%)")
        print(f"   Target range: {y.min()} to {y.max()} wickets")
        print(f"   Mean wickets: {y.mean():.2f}")

        # Wicket distribution
        print(f"\n   📊 Wicket Distribution:")
        for w in range(int(y.max()) + 1):
            count = (y == w).sum()
            pct = count / len(y) * 100
            bar = "█" * int(pct / 2)
            print(f"      {w} wickets: {count:4d} ({pct:5.1f}%) {bar}")

        # ── STEP 4: Model Training ────────────────────────────────────
        print("\n── Step 4: Model Training ──")
        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=3, random_state=42, n_jobs=1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                min_samples_split=5, random_state=42
            ),
            "XGBoost": XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, n_jobs=1
            ),
        }

        best_mae = float("inf")
        best_model_name = ""

        # ── STEP 5: Evaluate ──────────────────────────────────────────
        print("\n── Step 5: Evaluate ──")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")

            print(f"\n   🔹 {name}:")
            print(f"      MAE:    {mae:.3f} wickets")
            print(f"      RMSE:   {rmse:.3f} wickets")
            print(f"      R²:     {r2:.4f}")
            print(f"      CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            if mae < best_mae:
                best_mae = mae
                best_model_name = name
                self.model = model
                self.mae = mae
                self.r2 = r2

        print(f"\n   🏆 Best Model: {best_model_name} (MAE: {best_mae:.3f} wickets)")

        if hasattr(self.model, "feature_importances_"):
            importances = pd.Series(
                self.model.feature_importances_, index=self.feature_columns
            ).sort_values(ascending=False)
            print(f"\n   📈 Top 5 Most Important Features:")
            for feat, imp in importances.head(5).items():
                print(f"      {feat}: {imp:.4f}")

        return self.mae

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  STEP 6: PREDICT                                               ║
    # ║  User input → preprocess → model predicts → return wickets     ║
    # ╚══════════════════════════════════════════════════════════════════╝
    def predict(self, bowler, team, opponent, venue, pitch_type,
                match_format, bowling_avg, bowling_sr, bowling_econ,
                bowling_type, is_home, opponent_strength, overs_bowled):
        """Step 6: Predict — returns predicted wickets with confidence interval."""
        # Step 6a: Create input DataFrame
        input_data = pd.DataFrame([{
            "bowler": bowler, "team": team, "opponent": opponent,
            "venue": venue, "pitch_type": pitch_type,
            "match_format": match_format, "bowling_avg": bowling_avg,
            "bowling_sr": bowling_sr, "bowling_econ": bowling_econ,
            "bowling_type": bowling_type, "is_home": is_home,
            "opponent_strength": opponent_strength, "overs_bowled": overs_bowled,
        }])

        # Step 6b: Preprocess
        data = self.prepare_features(input_data)
        X = data[self.feature_columns]

        # Step 6c: Predict
        predicted_wickets = max(0, round(self.model.predict(X)[0], 1))

        # Step 6d: Confidence interval
        if hasattr(self.model, "estimators_"):
            if hasattr(self.model.estimators_[0], "predict"):
                tree_preds = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
            else:
                tree_preds = np.array([tree[0].predict(X)[0] for tree in self.model.estimators_])
            low = max(0, round(np.percentile(tree_preds, 25), 1))
            high = max(0, round(np.percentile(tree_preds, 75), 1))
        else:
            low = max(0, round(predicted_wickets - self.mae, 1))
            high = round(predicted_wickets + self.mae, 1)

        return predicted_wickets, (low, high)

    def save(self, path):
        """Save trained model to .pkl file."""
        joblib.dump({
            "model": self.model, "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns, "mae": self.mae, "r2": self.r2,
        }, path)
        print(f"💾 Wickets model saved to {path}")

    def load(self, path):
        """Load trained model from .pkl file."""
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoders = data["label_encoders"]
        self.feature_columns = data["feature_columns"]
        self.mae = data["mae"]
        self.r2 = data["r2"]
        print(f"📂 Wickets model loaded from {path}")
