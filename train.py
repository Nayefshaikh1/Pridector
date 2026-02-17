"""
Training Pipeline - Generates data, trains all models, saves them.
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

    print("\n🔧 STEP 1: GENERATING CRICKET DATA")
    generate_data()

    print("\n📂 STEP 2: LOADING DATA")
    matches_df = pd.read_csv(os.path.join(data_dir, "matches.csv"))
    batting_df = pd.read_csv(os.path.join(data_dir, "batting.csv"))
    bowling_df = pd.read_csv(os.path.join(data_dir, "bowling.csv"))

    match_predictor = MatchPredictor()
    match_predictor.train(matches_df)
    match_predictor.save(os.path.join(models_dir, "match_model.pkl"))

    runs_predictor = RunsPredictor()
    runs_predictor.train(batting_df)
    runs_predictor.save(os.path.join(models_dir, "runs_model.pkl"))

    wickets_predictor = WicketsPredictor()
    wickets_predictor.train(bowling_df)
    wickets_predictor.save(os.path.join(models_dir, "wickets_model.pkl"))

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"  Match Predictor:   Accuracy = {match_predictor.accuracy*100:.1f}%")
    print(f"  Runs Predictor:    MAE = {runs_predictor.mae:.2f} runs")
    print(f"  Wickets Predictor: MAE = {wickets_predictor.mae:.3f} wickets")
    print(f"\n  Run the app: streamlit run app.py")

if __name__ == "__main__":
    main()
