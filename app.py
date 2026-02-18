"""
Cricket Prediction System - Streamlit Web App
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.match_predictor import MatchPredictor
from models.runs_predictor import RunsPredictor
from models.wickets_predictor import WicketsPredictor
from data.generate_data import TEAMS, VENUES, BATSMEN, BOWLERS, TEAM_STRENGTH

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cricket Prediction AI", page_icon="🏏", layout="wide",
    initial_sidebar_state="expanded")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a3e 0%, #0d0d2b 100%); }
h1 { background: linear-gradient(90deg, #FF6B35, #F7C948, #FF6B35);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-weight: 900 !important; font-size: 2.5rem !important; }
h2 { color: #F7C948 !important; font-weight: 700 !important; }
h3 { color: #8B5CF6 !important; }
.stMetric { background: rgba(255,255,255,0.05); border-radius: 16px;
            padding: 20px; border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px); }
.stMetric label { color: #94a3b8 !important; }
.stMetric [data-testid="stMetricValue"] { color: #F7C948 !important; font-weight: 700 !important; }
div[data-testid="stRadio"] label { color: #e2e8f0 !important; }
.stSelectbox label, .stSlider label, .stNumberInput label { color: #cbd5e1 !important; }
.prediction-card { background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(59,130,246,0.2));
    border-radius: 20px; padding: 30px; margin: 15px 0; border: 1px solid rgba(139,92,246,0.3);
    backdrop-filter: blur(20px); }
.winner-badge { background: linear-gradient(135deg, #FF6B35, #F7C948);
    color: #1a1a2e; padding: 12px 28px; border-radius: 50px;
    font-weight: 800; font-size: 1.3rem; display: inline-block;
    box-shadow: 0 8px 32px rgba(255,107,53,0.4); }
.team-prob { background: rgba(255,255,255,0.08); border-radius: 12px;
    padding: 15px 25px; margin: 8px 0; color: #e2e8f0; }
.glow { text-shadow: 0 0 20px rgba(247,201,72,0.5); }
</style>
""", unsafe_allow_html=True)

# ─── Load or Train Models ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_dir, "models")
    data_dir = os.path.join(project_dir, "data")
    mp, rp, wp = MatchPredictor(), RunsPredictor(), WicketsPredictor()
    try:
        mp.load(os.path.join(models_dir, "match_model.pkl"))
        rp.load(os.path.join(models_dir, "runs_model.pkl"))
        wp.load(os.path.join(models_dir, "wickets_model.pkl"))
        return mp, rp, wp, True
    except Exception:
        # Auto-train if models not found (for cloud deployment)
        from data.generate_data import main as generate_data
        generate_data()
        matches_df = pd.read_csv(os.path.join(data_dir, "matches.csv"))
        batting_df = pd.read_csv(os.path.join(data_dir, "batting.csv"))
        bowling_df = pd.read_csv(os.path.join(data_dir, "bowling.csv"))
        mp.train(matches_df)
        mp.save(os.path.join(models_dir, "match_model.pkl"))
        rp.train(batting_df)
        rp.save(os.path.join(models_dir, "runs_model.pkl"))
        wp.train(bowling_df)
        wp.save(os.path.join(models_dir, "wickets_model.pkl"))
        return mp, rp, wp, True

match_pred, runs_pred, wickets_pred, models_loaded = load_models()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏏 Cricket AI")
    st.markdown("---")
    page = st.radio("Navigate", ["🏆 Match Prediction", "🏏 Runs Prediction",
                                   "🎯 Wickets Prediction", "📊 Data Explorer"],
                    label_visibility="collapsed")
    st.markdown("---")
    if models_loaded:
        st.success("✅ Models loaded!")
        st.metric("Match Accuracy", f"{match_pred.accuracy*100:.1f}%")
        st.metric("Runs MAE", f"{runs_pred.mae:.1f} runs")
        st.metric("Wickets MAE", f"{wickets_pred.mae:.2f}")
    st.markdown("---")
    st.caption("Built with Scikit-learn, XGBoost & Streamlit")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Match Prediction
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏆 Match Prediction":
    st.markdown("# 🏆 Match Winner Prediction")
    st.markdown("*Predict which team will win based on match conditions*")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("🏏 Team 1", TEAMS, index=0, key="mt1")
    with col2:
        team2_opts = [t for t in TEAMS if t != team1]
        team2 = st.selectbox("🏏 Team 2", team2_opts, index=0, key="mt2")
    col3, col4, col5 = st.columns(3)
    with col3:
        venue = st.selectbox("🏟️ Venue", list(VENUES.keys()), key="mv")
    with col4:
        match_format = st.selectbox("📋 Format", ["ODI", "T20"], key="mf")
    with col5:
        toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2], key="mtw")
    toss_decision = st.selectbox("🪙 Toss Decision", ["bat", "field"], key="mtd")

    venue_info = VENUES[venue]
    country_map = {"India":"India","Australia":"Australia","England":"England",
        "South Africa":"South Africa","New Zealand":"New Zealand","Pakistan":"Pakistan",
        "Sri Lanka":"Sri Lanka","Bangladesh":"Bangladesh","West Indies":"West Indies","UAE":"Pakistan"}
    home_team = country_map.get(venue_info["country"], "")
    t1_home = 1 if team1 == home_team else 0
    t2_home = 1 if team2 == home_team else 0

    if st.button("🔮 Predict Match Winner", use_container_width=True, type="primary"):
        winner, probs = match_pred.predict(team1, team2, venue, venue_info["pitch"],
            match_format, toss_winner, toss_decision, t1_home, t2_home,
            TEAM_STRENGTH[team1], TEAM_STRENGTH[team2])
        st.markdown("---")
        st.markdown(f'<div class="prediction-card"><div class="winner-badge glow">🏆 {winner} WINS!</div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"🏏 {team1}", f"{probs.get(team1, 0):.1f}%")
        c2.metric(f"🏏 {team2}", f"{probs.get(team2, 0):.1f}%")
        c3.metric("🏟️ Pitch", venue_info["pitch"].title())
        fig = go.Figure(go.Bar(x=[team1, team2], y=[probs.get(team1,0), probs.get(team2,0)],
            marker=dict(color=["#FF6B35","#8B5CF6"], line=dict(width=0)),
            text=[f"{probs.get(team1,0):.1f}%", f"{probs.get(team2,0):.1f}%"], textposition="auto"))
        fig.update_layout(title="Win Probability", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"), yaxis_title="Probability (%)")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Runs Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏏 Runs Prediction":
    st.markdown("# 🏏 Player Runs Prediction")
    st.markdown("*Predict how many runs a batsman will score*")
    st.markdown("---")
    team = st.selectbox("Select Team", TEAMS, key="rt")
    batsmen_list = BATSMEN.get(team, [])
    batsman_names = [b["name"] for b in batsmen_list]
    batsman_name = st.selectbox("Select Batsman", batsman_names, key="rb")
    batsman_info = next((b for b in batsmen_list if b["name"] == batsman_name), batsmen_list[0])
    col1, col2 = st.columns(2)
    with col1:
        opponent = st.selectbox("Opponent", [t for t in TEAMS if t != team], key="ro")
        venue = st.selectbox("Venue", list(VENUES.keys()), key="rv")
    with col2:
        match_format = st.selectbox("Format", ["ODI", "T20"], key="rf")
        batting_position = st.slider("Batting Position", 1, 8, 3, key="rp")
    venue_info = VENUES[venue]
    country_map = {"India":"India","Australia":"Australia","England":"England",
        "South Africa":"South Africa","New Zealand":"New Zealand","Pakistan":"Pakistan",
        "Sri Lanka":"Sri Lanka","Bangladesh":"Bangladesh","West Indies":"West Indies","UAE":"Pakistan"}
    is_home = 1 if country_map.get(venue_info["country"]) == team else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Career Average", f"{batsman_info['avg']}")
    c2.metric("Strike Rate", f"{batsman_info['sr']}")
    c3.metric("Home Ground", "Yes ✅" if is_home else "No ❌")

    if st.button("🔮 Predict Runs", use_container_width=True, type="primary"):
        runs, (low, high) = runs_pred.predict(batsman_name, team, opponent, venue,
            venue_info["pitch"], match_format, batsman_info["avg"], batsman_info["sr"],
            batsman_info["style"], batting_position, is_home, TEAM_STRENGTH[opponent])
        st.markdown("---")
        st.markdown(f'<div class="prediction-card"><div class="winner-badge glow">🏏 {runs} RUNS</div><br><p style="color:#94a3b8;margin-top:10px;">Expected Range: {low} - {high} runs</p></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted", f"{runs} runs")
        c2.metric("Low Estimate", f"{low} runs")
        c3.metric("High Estimate", f"{high} runs")
        fig = go.Figure()
        fig.add_trace(go.Indicator(mode="gauge+number", value=runs,
            title={"text": f"{batsman_name} - Predicted Runs"},
            gauge={"axis":{"range":[0, 150 if match_format=="T20" else 200]},
                   "bar":{"color":"#FF6B35"},
                   "steps":[{"range":[0,low],"color":"rgba(139,92,246,0.2)"},
                            {"range":[low,high],"color":"rgba(247,201,72,0.3)"},
                            {"range":[high,150 if match_format=="T20" else 200],"color":"rgba(139,92,246,0.1)"}]}))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"), height=350)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Wickets Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Wickets Prediction":
    st.markdown("# 🎯 Player Wickets Prediction")
    st.markdown("*Predict how many wickets a bowler will take*")
    st.markdown("---")
    team = st.selectbox("Select Team", TEAMS, key="wt")
    bowlers_list = BOWLERS.get(team, [])
    bowler_names = [b["name"] for b in bowlers_list]
    bowler_name = st.selectbox("Select Bowler", bowler_names, key="wb")
    bowler_info = next((b for b in bowlers_list if b["name"] == bowler_name), bowlers_list[0])
    col1, col2 = st.columns(2)
    with col1:
        opponent = st.selectbox("Opponent", [t for t in TEAMS if t != team], key="wo")
        venue = st.selectbox("Venue", list(VENUES.keys()), key="wv")
    with col2:
        match_format = st.selectbox("Format", ["ODI", "T20"], key="wf")
        overs = st.slider("Overs to Bowl", 1, 10 if match_format=="ODI" else 4,
                          6 if match_format=="ODI" else 4, key="wov")
    venue_info = VENUES[venue]
    country_map = {"India":"India","Australia":"Australia","England":"England",
        "South Africa":"South Africa","New Zealand":"New Zealand","Pakistan":"Pakistan",
        "Sri Lanka":"Sri Lanka","Bangladesh":"Bangladesh","West Indies":"West Indies","UAE":"Pakistan"}
    is_home = 1 if country_map.get(venue_info["country"]) == team else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Bowling Avg", f"{bowler_info['avg']}")
    c2.metric("Economy", f"{bowler_info['econ']}")
    c3.metric("Type", bowler_info["type"].title())

    if st.button("🔮 Predict Wickets", use_container_width=True, type="primary"):
        wickets, (low, high) = wickets_pred.predict(bowler_name, team, opponent, venue,
            venue_info["pitch"], match_format, bowler_info["avg"], bowler_info["sr"],
            bowler_info["econ"], bowler_info["type"], is_home, TEAM_STRENGTH[opponent], overs)
        st.markdown("---")
        st.markdown(f'<div class="prediction-card"><div class="winner-badge glow">🎯 {wickets} WICKETS</div><br><p style="color:#94a3b8;margin-top:10px;">Expected Range: {low} - {high} wickets</p></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted", f"{wickets} wickets")
        c2.metric("Low Estimate", f"{low}")
        c3.metric("High Estimate", f"{high}")
        fig = go.Figure()
        fig.add_trace(go.Indicator(mode="gauge+number", value=float(wickets),
            title={"text": f"{bowler_name} - Predicted Wickets"},
            gauge={"axis":{"range":[0,7]}, "bar":{"color":"#8B5CF6"},
                   "steps":[{"range":[0,1],"color":"rgba(59,130,246,0.2)"},
                            {"range":[1,3],"color":"rgba(247,201,72,0.2)"},
                            {"range":[3,5],"color":"rgba(255,107,53,0.2)"},
                            {"range":[5,7],"color":"rgba(239,68,68,0.3)"}]}))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"), height=350)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown("# 📊 Data Explorer")
    st.markdown("---")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    tab1, tab2, tab3 = st.tabs(["🏆 Matches", "🏏 Batting", "🎯 Bowling"])
    try:
        with tab1:
            df = pd.read_csv(os.path.join(data_dir, "matches.csv"))
            st.dataframe(df.head(50), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                win_counts = df["winner"].value_counts().head(10)
                fig = px.bar(x=win_counts.index, y=win_counts.values, title="Top Winners",
                    labels={"x":"Team","y":"Wins"}, color=win_counts.values,
                    color_continuous_scale="Viridis")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fmt = df["match_format"].value_counts()
                fig = px.pie(names=fmt.index, values=fmt.values, title="Format Distribution",
                    color_discrete_sequence=["#FF6B35","#8B5CF6"])
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            df = pd.read_csv(os.path.join(data_dir, "batting.csv"))
            st.dataframe(df.head(50), use_container_width=True)
            top = df.groupby("batsman")["runs_scored"].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=top.index, y=top.values, title="Top Batsmen by Avg Runs",
                labels={"x":"Batsman","y":"Avg Runs"}, color=top.values,
                color_continuous_scale="Magma")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            df = pd.read_csv(os.path.join(data_dir, "bowling.csv"))
            st.dataframe(df.head(50), use_container_width=True)
            top = df.groupby("bowler")["wickets_taken"].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=top.index, y=top.values, title="Top Bowlers by Avg Wickets",
                labels={"x":"Bowler","y":"Avg Wickets"}, color=top.values,
                color_continuous_scale="Plasma")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.warning("Data files not found. Run `python train.py` first!")
