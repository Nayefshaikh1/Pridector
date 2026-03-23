"""
Cricket Player Runs Prediction Model
=====================================
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


class RunsPredictor:
    """Predicts runs scored by a batsman in a match."""

    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.mae = 0
        self.r2 = 0
        self.model_name = "Player Runs Predictor"

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  STEP 2: DATA PREPROCESSING                                    ║
    # ║  - Label Encoding: Convert text (names, teams) to numbers      ║
    # ║  - Feature Engineering: Create new features from existing data ║
    # ║  - Feature Selection: Choose the 13 most useful features       ║
    # ╚══════════════════════════════════════════════════════════════════╝
    def prepare_features(self, df):
        """
        Step 2: Data Preprocessing
        - Converts categorical text data to numerical values using LabelEncoder
        - Engineers new features like avg_sr_product
        - Selects 13 features for the model
        """
        data = df.copy()

        # Step 2a: LABEL ENCODING
        # ML models only understand numbers, not text
        # So we convert: "Virat Kohli" → 15, "India" → 0, "ODI" → 1, etc.
        categorical_cols = ["batsman", "team", "opponent", "venue",
                            "pitch_type", "match_format", "batting_style"]

        for col in categorical_cols:
            if col not in self.label_encoders:
                # First time: fit the encoder on training data
                self.label_encoders[col] = LabelEncoder()
                data[f"{col}_encoded"] = self.label_encoders[col].fit_transform(data[col])
            else:
                # Prediction time: use the already-fitted encoder
                known_classes = set(self.label_encoders[col].classes_)
                data[col] = data[col].apply(lambda x: x if x in known_classes else self.label_encoders[col].classes_[0])
                data[f"{col}_encoded"] = self.label_encoders[col].transform(data[col])

        # Step 2b: FEATURE ENGINEERING
        # Create a new feature by combining batting_avg and strike_rate
        # Higher value = better batsman → likely to score more runs
        data["avg_sr_product"] = data["batting_avg"] * data["strike_rate"] / 100

        # Step 2c: FEATURE SELECTION
        # Choose the 13 most important features for prediction
        self.feature_columns = [
            "batsman_encoded",       # Which batsman (encoded as number)
            "team_encoded",          # Which team the batsman plays for
            "opponent_encoded",      # Which team they're playing against
            "venue_encoded",         # Which ground/stadium
            "pitch_type_encoded",    # batting/pace/spin/balanced pitch
            "match_format_encoded",  # ODI or T20
            "batting_avg",           # Career batting average (e.g., 59.1)
            "strike_rate",           # Career strike rate (e.g., 93.2)
            "batting_style_encoded", # Right-hand or Left-hand
            "batting_position",      # Position in batting order (1-8)
            "is_home",               # Playing at home ground? (0 or 1)
            "opponent_strength",     # How strong is the opponent (0-100)
            "avg_sr_product",        # Engineered: avg × sr / 100
        ]

        return data

    def train(self, df):
        """
        Executes Steps 2-5 of the ML pipeline:
        Step 2: Data Preprocessing (via prepare_features)
        Step 3: Split Data (80/20)
        Step 4: Model Training (3 algorithms)
        Step 5: Evaluate (MAE, RMSE, R², Cross-Validation)
        """
        print("\n" + "=" * 60)
        print("🏏 PLAYER RUNS PREDICTION MODEL")
        print("=" * 60)

        # ── STEP 2: Data Preprocessing ────────────────────────────────
        print("\n── Step 2: Data Preprocessing ──")
        data = self.prepare_features(df)
        print(f"   ✅ Encoded {len(self.label_encoders)} categorical columns")
        print(f"   ✅ Engineered feature: avg_sr_product")
        print(f"   ✅ Selected {len(self.feature_columns)} features")

        # ── STEP 3: Split Data ────────────────────────────────────────
        # X = input features (13 columns), y = target (runs_scored)
        print("\n── Step 3: Split Data ──")
        X = data[self.feature_columns]
        y = data["runs_scored"]

        # Split: 80% for training, 20% for testing
        # random_state=42 ensures same split every time (reproducibility)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   Total records:   {len(X)} ")
        print(f"   Training set:    {len(X_train)} records (80%)")
        print(f"   Testing set:     {len(X_test)} records (20%)")
        print(f"   Target range:    {y.min()} to {y.max()} runs")
        print(f"   Mean runs:       {y.mean():.1f}")
        print(f"   Median runs:     {y.median():.1f}")

        # ── STEP 4: Model Training ────────────────────────────────────
        # Train 3 different ML algorithms and compare them
        print("\n── Step 4: Model Training ──")
        print("   Training 3 algorithms: Random Forest, Gradient Boosting, XGBoost")

        models = {
            # Algorithm 1: Random Forest (Bagging - 100 parallel trees)
            "Random Forest": RandomForestRegressor(
                n_estimators=100,      # Build 100 decision trees
                max_depth=12,          # Each tree max 12 levels deep
                min_samples_split=5,   # Need 5+ samples to split a node
                min_samples_leaf=3,    # Each leaf needs 3+ samples
                random_state=42,       # Reproducibility
                n_jobs=1               # Use 1 CPU core
            ),
            # Algorithm 2: Gradient Boosting (Boosting - 100 sequential trees)
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100,      # 100 sequential trees
                max_depth=6,           # Shallow trees (boosting works best with weak learners)
                learning_rate=0.1,     # Small steps = better generalization
                min_samples_split=5,
                random_state=42
            ),
            # Algorithm 3: XGBoost (Optimized Boosting with regularization)
            "XGBoost": XGBRegressor(
                n_estimators=100,      # 100 boosting rounds
                max_depth=6,           # Tree depth
                learning_rate=0.1,     # Step size
                random_state=42,
                n_jobs=1
            ),
        }

        best_mae = float("inf")
        best_model_name = ""

        # ── STEP 5: Evaluate ──────────────────────────────────────────
        print("\n── Step 5: Evaluate ──")
        print("   Testing each model on the 20% test data it has NEVER seen:\n")

        for name, model in models.items():
            # Step 4: Train the model (model learns patterns from training data)
            model.fit(X_train, y_train)
            # model.fit() = "learn the relationship between features (X) and runs (y)"

            # Step 5: Evaluate the model on TEST data (data it never saw during training)
            y_pred = model.predict(X_test)
            # y_pred = model's predictions for the test data
            # y_test = actual correct answers

            # Metric 1: MAE (Mean Absolute Error)
            # Average of |actual - predicted| for all test samples
            # Example: |45 - 38| + |23 - 30| + ... / 1000 = 18.58 runs
            mae = mean_absolute_error(y_test, y_pred)

            # Metric 2: RMSE (Root Mean Squared Error)
            # Like MAE but penalizes large errors more heavily
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Metric 3: R² Score (Coefficient of Determination)
            # How much variance in runs the model explains (0 to 1, higher = better)
            r2 = r2_score(y_test, y_pred)

            # Metric 4: Cross-Validation (5-Fold)
            # Split data 5 different ways, train and test on each → average score
            # This ensures our result is not a fluke
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")

            print(f"   🔹 {name}:")
            print(f"      MAE:    {mae:.2f} runs (avg prediction error)")
            print(f"      RMSE:   {rmse:.2f} runs (penalizes big errors)")
            print(f"      R²:     {r2:.4f} (variance explained)")
            print(f"      CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} (5-fold avg)")

            # Auto-select the model with LOWEST MAE (fewest prediction errors)
            if mae < best_mae:
                best_mae = mae
                best_model_name = name
                self.model = model
                self.mae = mae
                self.r2 = r2

        print(f"\n   🏆 Best Model: {best_model_name} (MAE: {best_mae:.2f} runs)")

        # Show which features are most important for prediction
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
    # ║  Takes user input → preprocesses → feeds to model → returns    ║
    # ║  predicted runs with confidence interval (low-high range)      ║
    # ╚══════════════════════════════════════════════════════════════════╝
    def predict(self, batsman, team, opponent, venue, pitch_type,
                match_format, batting_avg, strike_rate, batting_style,
                batting_position, is_home, opponent_strength):
        """
        Step 6: Predict
        Takes user input from the Streamlit app and returns:
        - predicted_runs: the model's best guess
        - (low, high): confidence interval from individual tree predictions
        """
        # Step 6a: Create a DataFrame from user's input
        input_data = pd.DataFrame([{
            "batsman": batsman, "team": team, "opponent": opponent,
            "venue": venue, "pitch_type": pitch_type,
            "match_format": match_format, "batting_avg": batting_avg,
            "strike_rate": strike_rate, "batting_style": batting_style,
            "batting_position": batting_position, "is_home": is_home,
            "opponent_strength": opponent_strength,
        }])

        # Step 6b: Preprocess the input (same encoding as training)
        data = self.prepare_features(input_data)
        X = data[self.feature_columns]

        # Step 6c: Feed input to the trained model → get prediction
        predicted_runs = max(0, int(round(self.model.predict(X)[0])))

        # Step 6d: Calculate confidence interval
        # Each tree in the forest gives a different prediction
        # We use the 25th and 75th percentile as the range
        if hasattr(self.model, "estimators_"):
            if hasattr(self.model.estimators_[0], "predict"):
                tree_preds = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
            else:
                tree_preds = np.array([tree[0].predict(X)[0] for tree in self.model.estimators_])
            low = max(0, int(np.percentile(tree_preds, 25)))   # 25th percentile
            high = max(0, int(np.percentile(tree_preds, 75)))  # 75th percentile
        else:
            low = max(0, predicted_runs - int(self.mae))
            high = predicted_runs + int(self.mae)

        # Return: predicted runs and confidence range
        return predicted_runs, (low, high)

    def save(self, path):
        """Save the trained model and encoders to a .pkl file."""
        joblib.dump({
            "model": self.model,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "mae": self.mae,
            "r2": self.r2,
        }, path)
        print(f"💾 Runs model saved to {path}")

    def load(self, path):
        """Load a previously trained model from a .pkl file."""
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoders = data["label_encoders"]
        self.feature_columns = data["feature_columns"]
        self.mae = data["mae"]
        self.r2 = data["r2"]
        print(f"📂 Runs model loaded from {path}")
