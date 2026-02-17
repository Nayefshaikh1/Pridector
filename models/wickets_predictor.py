"""
Cricket Player Wickets Prediction Model
========================================
Predicts the number of wickets a bowler will take based on
player stats, opponent, venue, and match conditions.
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

    def prepare_features(self, df):
        """Prepare features for training/prediction."""
        data = df.copy()

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

        # Feature engineering
        data["avg_econ_ratio"] = data["bowling_avg"] / (data["bowling_econ"] + 0.1)
        data["sr_econ_ratio"] = data["bowling_sr"] / (data["bowling_econ"] + 0.1)

        # Is pitch favorable for this bowler type?
        data["pitch_favorable"] = 0
        data.loc[(data["pitch_type"] == "pace") & (data["bowling_type"] == "fast"), "pitch_favorable"] = 1
        data.loc[(data["pitch_type"] == "spin") & (data["bowling_type"] == "spin"), "pitch_favorable"] = 1

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
        """Train the wickets prediction model."""
        print("\n" + "=" * 60)
        print("🏏 TRAINING PLAYER WICKETS PREDICTION MODEL")
        print("=" * 60)

        data = self.prepare_features(df)

        X = data[self.feature_columns]
        y = data["wickets_taken"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n📊 Dataset: {len(X_train)} training, {len(X_test)} testing samples")
        print(f"📊 Target range: {y.min()} to {y.max()} wickets")
        print(f"📊 Mean wickets: {y.mean():.2f}, Median: {y.median():.1f}")

        # Wicket distribution
        print(f"\n📊 Wicket Distribution:")
        for w in range(int(y.max()) + 1):
            count = (y == w).sum()
            pct = count / len(y) * 100
            bar = "█" * int(pct / 2)
            print(f"   {w} wickets: {count:4d} ({pct:5.1f}%) {bar}")

        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_split=5,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                min_samples_split=5, random_state=42
            ),
            "XGBoost": XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42
            ),
        }

        best_mae = float("inf")
        best_model_name = ""

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")

            print(f"\n🔹 {name}:")
            print(f"   MAE:   {mae:.3f} wickets")
            print(f"   RMSE:  {rmse:.3f} wickets")
            print(f"   R²:    {r2:.4f}")
            print(f"   CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            if mae < best_mae:
                best_mae = mae
                best_model_name = name
                self.model = model
                self.mae = mae
                self.r2 = r2

        print(f"\n🏆 Best Model: {best_model_name} (MAE: {best_mae:.3f} wickets)")

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            importances = pd.Series(
                self.model.feature_importances_, index=self.feature_columns
            ).sort_values(ascending=False)
            print(f"\n📈 Top Feature Importances:")
            for feat, imp in importances.head(5).items():
                print(f"   {feat}: {imp:.4f}")

        return self.mae

    def predict(self, bowler, team, opponent, venue, pitch_type,
                match_format, bowling_avg, bowling_sr, bowling_econ,
                bowling_type, is_home, opponent_strength, overs_bowled):
        """Predict wickets for a bowler."""
        input_data = pd.DataFrame([{
            "bowler": bowler, "team": team, "opponent": opponent,
            "venue": venue, "pitch_type": pitch_type,
            "match_format": match_format, "bowling_avg": bowling_avg,
            "bowling_sr": bowling_sr, "bowling_econ": bowling_econ,
            "bowling_type": bowling_type, "is_home": is_home,
            "opponent_strength": opponent_strength, "overs_bowled": overs_bowled,
        }])

        data = self.prepare_features(input_data)
        X = data[self.feature_columns]

        predicted_wickets = max(0, round(self.model.predict(X)[0], 1))

        # Generate prediction range
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
        """Save the model and encoders."""
        joblib.dump({
            "model": self.model,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "mae": self.mae,
            "r2": self.r2,
        }, path)
        print(f"💾 Wickets model saved to {path}")

    def load(self, path):
        """Load the model and encoders."""
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoders = data["label_encoders"]
        self.feature_columns = data["feature_columns"]
        self.mae = data["mae"]
        self.r2 = data["r2"]
        print(f"📂 Wickets model loaded from {path}")
