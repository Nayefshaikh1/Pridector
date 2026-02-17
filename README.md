# 🏏 Cricket Prediction System - AI/ML Project

An AI-powered cricket prediction system that uses Machine Learning to predict:

1. **Match Winner** - Which team will win based on teams, venue, pitch, toss, etc.
2. **Player Runs** - How many runs a batsman will score
3. **Player Wickets** - How many wickets a bowler will take

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-learn** - Random Forest, Gradient Boosting
- **XGBoost** - Extreme Gradient Boosting
- **Pandas & NumPy** - Data processing
- **Streamlit** - Interactive web application
- **Plotly** - Data visualizations

## 🚀 Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python train.py
```

### 3. Launch Web App
```bash
streamlit run app.py
```

## 📁 Project Structure

```
cricket-prediction/
├── data/
│   ├── generate_data.py      # Synthetic data generator
│   ├── matches.csv            # Generated match data
│   ├── batting.csv            # Generated batting data
│   └── bowling.csv            # Generated bowling data
├── models/
│   ├── match_predictor.py     # Match winner ML model
│   ├── runs_predictor.py      # Runs prediction ML model
│   ├── wickets_predictor.py   # Wickets prediction ML model
│   ├── match_model.pkl        # Saved match model
│   ├── runs_model.pkl         # Saved runs model
│   └── wickets_model.pkl      # Saved wickets model
├── app.py                     # Streamlit web app
├── train.py                   # Training pipeline
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 📊 Models & Features

### Match Prediction (Classification)
- **Features**: Team strengths, venue, pitch type, toss, home advantage
- **Models**: Random Forest, Gradient Boosting, XGBoost
- **Best is auto-selected** based on test accuracy

### Runs Prediction (Regression)
- **Features**: Player average, strike rate, opponent, venue, pitch, format
- **Models**: Random Forest, Gradient Boosting, XGBoost
- **Outputs**: Predicted runs + confidence interval

### Wickets Prediction (Regression)
- **Features**: Bowling average, economy, bowling type, pitch favorability
- **Models**: Random Forest, Gradient Boosting, XGBoost
- **Outputs**: Predicted wickets + confidence interval

## 🏏 Teams Covered
India, Australia, England, South Africa, New Zealand, Pakistan, Sri Lanka, Bangladesh, West Indies, Afghanistan

## 👨‍💻 Author
Cricket Prediction AI - ML Project
