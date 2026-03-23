"""
Cricket Prediction - ML Training Pipeline
==========================================
This script executes the complete ML pipeline:
  Step 1: Collect Dataset (generate synthetic cricket data)
  Step 2: Data Preprocessing (handled inside each model's prepare_features())
  Step 3: Split Data (80% training, 20% testing)
  Step 4: Model Training (Random Forest, Gradient Boosting, XGBoost)
  Step 5: Evaluate (MAE, RMSE, R², Cross-Validation, auto-select best model)
  Step 6: Predict (available via the Streamlit web app - app.py)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_data import main as generate_data
from models.match_predictor import MatchPredictor
from models.runs_predictor import RunsPredictor
from models.wickets_predictor import WicketsPredictor
import pandas as pd

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")
    models_dir = os.path.join(project_dir, "models")

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  STEP 1: COLLECT DATASET                                       ║
    # ║  Generate synthetic cricket data based on real player stats     ║
    # ║  Output: matches.csv (3000), batting.csv (5000), bowling.csv   ║
    # ╚══════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 60)
    print("STEP 1: COLLECT DATASET - Generating Cricket Data")
    print("=" * 60)
    generate_data()
    # After this step, 3 CSV files are created in the data/ folder:
    #   - matches.csv  → 3,000 match records with winner labels
    #   - batting.csv  → 5,000 batting performance records
    #   - bowling.csv  → 5,000 bowling performance records

    # Load the generated datasets into pandas DataFrames
    print("\n📂 Loading generated datasets...")
    matches_df = pd.read_csv(os.path.join(data_dir, "matches.csv"))
    batting_df = pd.read_csv(os.path.join(data_dir, "batting.csv"))
    bowling_df = pd.read_csv(os.path.join(data_dir, "bowling.csv"))
    print(f"   Matches: {matches_df.shape[0]} records, {matches_df.shape[1]} columns")
    print(f"   Batting: {batting_df.shape[0]} records, {batting_df.shape[1]} columns")
    print(f"   Bowling: {bowling_df.shape[0]} records, {bowling_df.shape[1]} columns")

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  STEPS 2-5: PREPROCESSING → SPLIT → TRAIN → EVALUATE          ║
    # ║  Each model's .train() method handles Steps 2 through 5:       ║
    # ║    Step 2: Data Preprocessing (LabelEncoder + Feature Eng.)    ║
    # ║    Step 3: Split (80% train / 20% test)                        ║
    # ║    Step 4: Train (fit 3 ML algorithms)                         ║
    # ║    Step 5: Evaluate (MAE, RMSE, R², Cross-Validation)          ║
    # ╚══════════════════════════════════════════════════════════════════╝

    # --- Match Winner Prediction Model (Classification) ---
    print("\n" + "=" * 60)
    print("TRAINING MODEL 1: Match Winner Prediction")
    print("=" * 60)
    match_predictor = MatchPredictor()
    match_predictor.train(matches_df)       # Steps 2-5 happen inside
    match_predictor.save(os.path.join(models_dir, "match_model.pkl"))

    # --- Player Runs Prediction Model (Regression) ---
    print("\n" + "=" * 60)
    print("TRAINING MODEL 2: Player Runs Prediction")
    print("=" * 60)
    runs_predictor = RunsPredictor()
    runs_predictor.train(batting_df)        # Steps 2-5 happen inside
    runs_predictor.save(os.path.join(models_dir, "runs_model.pkl"))

    # --- Player Wickets Prediction Model (Regression) ---
    print("\n" + "=" * 60)
    print("TRAINING MODEL 3: Player Wickets Prediction")
    print("=" * 60)
    wickets_predictor = WicketsPredictor()
    wickets_predictor.train(bowling_df)     # Steps 2-5 happen inside
    wickets_predictor.save(os.path.join(models_dir, "wickets_model.pkl"))

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  RESULTS SUMMARY                                                ║
    # ║  Step 6 (Predict) is handled by app.py (Streamlit web app)     ║
    # ╚══════════════════════════════════════════════════════════════════╝
    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)
    print(f"  Model 1 - Match Predictor:   Accuracy = {match_predictor.accuracy*100:.1f}%")
    print(f"  Model 2 - Runs Predictor:    MAE = {runs_predictor.mae:.2f} runs")
    print(f"  Model 3 - Wickets Predictor: MAE = {wickets_predictor.mae:.3f} wickets")
    print(f"\n  Step 6 (Predict): Run 'streamlit run app.py' to make predictions")

if __name__ == "__main__":
    main()
