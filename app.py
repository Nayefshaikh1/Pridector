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
from data.generate_ipl_data import (IPL_TEAMS, IPL_VENUES, IPL_PLAYERS,
                                     IPL_TEAM_STRENGTH, IPL_TEAM_SHORT)

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

# Auto-generate IPL data if not present
@st.cache_resource
def ensure_ipl_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    ipl_path = os.path.join(data_dir, "ipl.csv")
    if not os.path.exists(ipl_path):
        from data.generate_ipl_data import generate_ipl_data
        ipl_df = generate_ipl_data(2000)
        ipl_df.to_csv(ipl_path, index=False)
    return True

ensure_ipl_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏏 Cricket AI")
    st.markdown("---")
    page = st.radio("Navigate", ["🏆 Match Prediction", "🏏 Runs Prediction",
                                   "🎯 Wickets Prediction", "📊 Data Explorer",
                                   "🏆 IPL Explorer", "🏏 IPL Prediction",
                                   "👥 IPL Teams", "📝 Add Data"],
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

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: IPL Match Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 IPL Prediction":
    st.markdown("# 🏆 IPL Match Prediction")
    st.markdown("*Predict which IPL team will win based on match conditions*")
    st.markdown("---")

    if not ipl_models_loaded or not ipl_match_pred:
        st.error("❌ IPL models not loaded. Please regenerate data and retrain.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            ipl_team1 = st.selectbox("🏏 Team 1", IPL_TEAMS, index=0, key="ipl_t1")
        with col2:
            ipl_team2_opts = [t for t in IPL_TEAMS if t != ipl_team1]
            ipl_team2 = st.selectbox("🏏 Team 2", ipl_team2_opts, index=0, key="ipl_t2")

        # Show team abbreviations
        st.markdown(f"**{IPL_TEAM_SHORT.get(ipl_team1, '')} vs {IPL_TEAM_SHORT.get(ipl_team2, '')}**")

        col3, col4 = st.columns(2)
        with col3:
            ipl_venue = st.selectbox("🏟️ Venue", list(IPL_VENUES.keys()), key="ipl_v")
        with col4:
            ipl_toss_winner = st.selectbox("🪙 Toss Winner", [ipl_team1, ipl_team2], key="ipl_tw")
        ipl_toss_decision = st.selectbox("🪙 Toss Decision", ["bat", "field"], key="ipl_td")

        ipl_venue_info = IPL_VENUES[ipl_venue]
        t1_home = 1 if ipl_venue_info["home_team"] == ipl_team1 else 0
        t2_home = 1 if ipl_venue_info["home_team"] == ipl_team2 else 0

        # Show venue info
        c1, c2, c3 = st.columns(3)
        c1.metric("🏟️ City", ipl_venue_info["city"])
        c2.metric("🏏 Pitch", ipl_venue_info["pitch"].title())
        home_text = "🏠 " + (ipl_team1 if t1_home else ipl_team2 if t2_home else "Neutral")
        c3.metric("Home Team", home_text)

        if st.button("🔮 Predict IPL Winner", use_container_width=True, type="primary"):
            winner, probs = ipl_match_pred.predict(
                ipl_team1, ipl_team2, ipl_venue, ipl_venue_info["pitch"],
                "IPL", ipl_toss_winner, ipl_toss_decision, t1_home, t2_home,
                IPL_TEAM_STRENGTH[ipl_team1], IPL_TEAM_STRENGTH[ipl_team2]
            )
            st.markdown("---")

            winner_short = IPL_TEAM_SHORT.get(winner, winner)
            st.markdown(f'<div class="prediction-card"><div class="winner-badge glow">🏆 {winner_short} WINS!</div><br><p style="color:#94a3b8;">{winner}</p></div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            t1_short = IPL_TEAM_SHORT.get(ipl_team1, ipl_team1)
            t2_short = IPL_TEAM_SHORT.get(ipl_team2, ipl_team2)
            c1.metric(f"🏏 {t1_short}", f"{probs.get(ipl_team1, 0):.1f}%")
            c2.metric(f"🏏 {t2_short}", f"{probs.get(ipl_team2, 0):.1f}%")
            c3.metric("🏟️ Pitch", ipl_venue_info["pitch"].title())

            # Win probability chart
            fig = go.Figure(go.Bar(
                x=[t1_short, t2_short],
                y=[probs.get(ipl_team1, 0), probs.get(ipl_team2, 0)],
                marker=dict(color=["#FFD700", "#8B5CF6"], line=dict(width=0)),
                text=[f"{probs.get(ipl_team1, 0):.1f}%", f"{probs.get(ipl_team2, 0):.1f}%"],
                textposition="auto"
            ))
            fig.update_layout(title="IPL Win Probability", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), yaxis_title="Probability (%)")
            st.plotly_chart(fig, use_container_width=True)

            # Head to head info
            st.markdown("### 📊 Match Factors")
            factors_col1, factors_col2 = st.columns(2)
            with factors_col1:
                st.markdown(f"**{t1_short} Strength:** {IPL_TEAM_STRENGTH[ipl_team1]}/100")
                st.markdown(f"**Home Advantage:** {'Yes ✅' if t1_home else 'No ❌'}")
            with factors_col2:
                st.markdown(f"**{t2_short} Strength:** {IPL_TEAM_STRENGTH[ipl_team2]}/100")
                st.markdown(f"**Home Advantage:** {'Yes ✅' if t2_home else 'No ❌'}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: IPL Player Stats
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏏 IPL Player Stats":
    st.markdown("# 🏏 IPL Player Predictions")
    st.markdown("*Predict runs and wickets for IPL players*")
    st.markdown("---")

    ipl_tab1, ipl_tab2 = st.tabs(["🏏 Runs Prediction", "🎯 Wickets Prediction"])

    with ipl_tab1:
        if not ipl_models_loaded or not ipl_runs_pred:
            st.error("❌ IPL runs model not loaded.")
        else:
            ipl_bat_team = st.selectbox("Select IPL Team", IPL_TEAMS, key="ipl_bt")
            batsmen_list = IPL_BATSMEN.get(ipl_bat_team, [])
            batsman_names = [b["name"] for b in batsmen_list]
            batsman_name = st.selectbox("Select Batsman", batsman_names, key="ipl_bn")
            batsman_info = next((b for b in batsmen_list if b["name"] == batsman_name), batsmen_list[0])

            col1, col2 = st.columns(2)
            with col1:
                ipl_bat_opp = st.selectbox("Opponent", [t for t in IPL_TEAMS if t != ipl_bat_team], key="ipl_bo")
                ipl_bat_venue = st.selectbox("Venue", list(IPL_VENUES.keys()), key="ipl_bv")
            with col2:
                ipl_bat_pos = st.slider("Batting Position", 1, 8, 3, key="ipl_bp")

            venue_info = IPL_VENUES[ipl_bat_venue]
            is_home = 1 if venue_info["home_team"] == ipl_bat_team else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("IPL Average", f"{batsman_info['avg']}")
            c2.metric("Strike Rate", f"{batsman_info['sr']}")
            c3.metric("Home Ground", "Yes ✅" if is_home else "No ❌")

            if st.button("🔮 Predict IPL Runs", use_container_width=True, type="primary", key="ipl_runs_btn"):
                runs, (low, high) = ipl_runs_pred.predict(
                    batsman_name, ipl_bat_team, ipl_bat_opp, ipl_bat_venue,
                    venue_info["pitch"], "IPL", batsman_info["avg"], batsman_info["sr"],
                    batsman_info["style"], ipl_bat_pos, is_home, IPL_TEAM_STRENGTH[ipl_bat_opp]
                )
                st.markdown("---")
                st.markdown(f'<div class="prediction-card"><div class="winner-badge glow">🏏 {runs} RUNS</div><br><p style="color:#94a3b8;margin-top:10px;">Expected Range: {low} - {high} runs</p></div>', unsafe_allow_html=True)

                fig = go.Figure()
                fig.add_trace(go.Indicator(mode="gauge+number", value=runs,
                    title={"text": f"{batsman_name} - IPL Predicted Runs"},
                    gauge={"axis":{"range":[0, 120]},
                           "bar":{"color":"#FFD700"},
                           "steps":[{"range":[0,low],"color":"rgba(139,92,246,0.2)"},
                                    {"range":[low,high],"color":"rgba(247,201,72,0.3)"},
                                    {"range":[high,120],"color":"rgba(139,92,246,0.1)"}]}))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"), height=350)
                st.plotly_chart(fig, use_container_width=True)

    with ipl_tab2:
        if not ipl_models_loaded or not ipl_wickets_pred:
            st.error("❌ IPL wickets model not loaded.")
        else:
            ipl_bowl_team = st.selectbox("Select IPL Team", IPL_TEAMS, key="ipl_wt")
            bowlers_list = IPL_BOWLERS.get(ipl_bowl_team, [])
            bowler_names = [b["name"] for b in bowlers_list]
            bowler_name = st.selectbox("Select Bowler", bowler_names, key="ipl_wn")
            bowler_info = next((b for b in bowlers_list if b["name"] == bowler_name), bowlers_list[0])

            col1, col2 = st.columns(2)
            with col1:
                ipl_bowl_opp = st.selectbox("Opponent", [t for t in IPL_TEAMS if t != ipl_bowl_team], key="ipl_wo")
                ipl_bowl_venue = st.selectbox("Venue", list(IPL_VENUES.keys()), key="ipl_wv")
            with col2:
                ipl_bowl_overs = st.slider("Overs to Bowl", 1, 4, 4, key="ipl_wov")

            venue_info = IPL_VENUES[ipl_bowl_venue]
            is_home = 1 if venue_info["home_team"] == ipl_bowl_team else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Bowling Avg", f"{bowler_info['avg']}")
            c2.metric("Economy", f"{bowler_info['econ']}")
            c3.metric("Type", bowler_info["type"].title())

            if st.button("🔮 Predict IPL Wickets", use_container_width=True, type="primary", key="ipl_wick_btn"):
                wickets, (low, high) = ipl_wickets_pred.predict(
                    bowler_name, ipl_bowl_team, ipl_bowl_opp, ipl_bowl_venue,
                    venue_info["pitch"], "IPL", bowler_info["avg"], bowler_info["sr"],
                    bowler_info["econ"], bowler_info["type"], is_home,
                    IPL_TEAM_STRENGTH[ipl_bowl_opp], ipl_bowl_overs
                )
                st.markdown("---")
                st.markdown(f'<div class="prediction-card"><div class="winner-badge glow">🎯 {wickets} WICKETS</div><br><p style="color:#94a3b8;margin-top:10px;">Expected Range: {low} - {high} wickets</p></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Add Data
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Add Data":
    st.markdown("# 📝 Add New Data")
    st.markdown("*Add new match, batting, or bowling records to improve predictions*")
    st.markdown("---")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    add_tab1, add_tab2, add_tab3 = st.tabs(["🏆 Add Match", "🏏 Add Batting", "🎯 Add Bowling"])

    # ─── TAB 1: Add Match Data ────────────────────────────────────────────────
    with add_tab1:
        st.markdown("### 🏆 Add Match Record")
        with st.form("match_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                m_team1 = st.selectbox("Team 1", TEAMS, key="am_t1")
                m_venue = st.selectbox("Venue", list(VENUES.keys()), key="am_v")
                m_toss_winner_option = st.selectbox("Toss Winner", ["Team 1", "Team 2"], key="am_tw")
                m_team1_score = st.number_input("Team 1 Score", min_value=0, max_value=500, value=250, key="am_s1")
            with c2:
                m_team2 = st.selectbox("Team 2", [t for t in TEAMS], key="am_t2")
                m_format = st.selectbox("Format", ["ODI", "T20"], key="am_f")
                m_toss_decision = st.selectbox("Toss Decision", ["bat", "field"], key="am_td")
                m_team2_score = st.number_input("Team 2 Score", min_value=0, max_value=500, value=230, key="am_s2")
            m_winner = st.selectbox("Match Winner", TEAMS, key="am_w")

            if st.form_submit_button("➕ Add Match Record", use_container_width=True, type="primary"):
                if m_team1 == m_team2:
                    st.error("❌ Team 1 and Team 2 cannot be the same!")
                elif m_winner not in [m_team1, m_team2]:
                    st.error("❌ Winner must be one of the two teams!")
                else:
                    venue_info = VENUES[m_venue]
                    country_map = {"India":"India","Australia":"Australia","England":"England",
                        "South Africa":"South Africa","New Zealand":"New Zealand","Pakistan":"Pakistan",
                        "Sri Lanka":"Sri Lanka","Bangladesh":"Bangladesh","West Indies":"West Indies","UAE":"Pakistan"}
                    home_team = country_map.get(venue_info["country"], "")
                    m_toss_w = m_team1 if m_toss_winner_option == "Team 1" else m_team2
                    try:
                        existing = pd.read_csv(os.path.join(data_dir, "matches.csv"))
                        new_id = existing["match_id"].max() + 1
                    except:
                        new_id = 1
                    new_row = pd.DataFrame([{
                        "match_id": new_id, "team1": m_team1, "team2": m_team2,
                        "venue": m_venue, "pitch_type": venue_info["pitch"],
                        "match_format": m_format, "toss_winner": m_toss_w,
                        "toss_decision": m_toss_decision,
                        "team1_home": 1 if m_team1 == home_team else 0,
                        "team2_home": 1 if m_team2 == home_team else 0,
                        "team1_strength": TEAM_STRENGTH[m_team1],
                        "team2_strength": TEAM_STRENGTH[m_team2],
                        "team1_score": m_team1_score, "team2_score": m_team2_score,
                        "winner": m_winner,
                    }])
                    filepath = os.path.join(data_dir, "matches.csv")
                    new_row.to_csv(filepath, mode="a", header=not os.path.exists(filepath), index=False)
                    st.success(f"✅ Match record added! {m_team1} vs {m_team2} → Winner: {m_winner}")

    # ─── TAB 2: Add Batting Data ──────────────────────────────────────────────
    with add_tab2:
        st.markdown("### 🏏 Add Batting Record")
        with st.form("batting_form", clear_on_submit=True):
            b_team = st.selectbox("Batsman's Team", TEAMS, key="ab_t")
            batsmen_list = BATSMEN.get(b_team, [])
            b_names = [b["name"] for b in batsmen_list]
            c1, c2 = st.columns(2)
            with c1:
                b_name = st.selectbox("Batsman", b_names, key="ab_n")
                b_opponent = st.selectbox("Opponent", [t for t in TEAMS if t != b_team], key="ab_o")
                b_venue = st.selectbox("Venue", list(VENUES.keys()), key="ab_v")
                b_format = st.selectbox("Format", ["ODI", "T20"], key="ab_f")
            with c2:
                b_runs = st.number_input("Runs Scored", min_value=0, max_value=300, value=45, key="ab_r")
                b_balls = st.number_input("Balls Faced", min_value=1, max_value=400, value=52, key="ab_bl")
                b_fours = st.number_input("Fours", min_value=0, max_value=40, value=4, key="ab_4")
                b_sixes = st.number_input("Sixes", min_value=0, max_value=20, value=1, key="ab_6")
            b_position = st.slider("Batting Position", 1, 8, 3, key="ab_p")
            b_not_out = st.checkbox("Not Out?", key="ab_no")

            if st.form_submit_button("➕ Add Batting Record", use_container_width=True, type="primary"):
                b_info = next((b for b in batsmen_list if b["name"] == b_name), batsmen_list[0])
                venue_info = VENUES[b_venue]
                country_map = {"India":"India","Australia":"Australia","England":"England",
                    "South Africa":"South Africa","New Zealand":"New Zealand","Pakistan":"Pakistan",
                    "Sri Lanka":"Sri Lanka","Bangladesh":"Bangladesh","West Indies":"West Indies"}
                is_home = 1 if country_map.get(venue_info["country"]) == b_team else 0
                try:
                    existing = pd.read_csv(os.path.join(data_dir, "batting.csv"))
                    new_id = existing["innings_id"].max() + 1
                except:
                    new_id = 1
                new_row = pd.DataFrame([{
                    "innings_id": new_id, "batsman": b_name, "team": b_team,
                    "opponent": b_opponent, "venue": b_venue,
                    "pitch_type": venue_info["pitch"], "match_format": b_format,
                    "batting_avg": b_info["avg"], "strike_rate": b_info["sr"],
                    "batting_style": b_info["style"], "batting_position": b_position,
                    "is_home": is_home, "opponent_strength": TEAM_STRENGTH[b_opponent],
                    "runs_scored": b_runs, "balls_faced": b_balls,
                    "fours": b_fours, "sixes": b_sixes, "not_out": 1 if b_not_out else 0,
                }])
                filepath = os.path.join(data_dir, "batting.csv")
                new_row.to_csv(filepath, mode="a", header=not os.path.exists(filepath), index=False)
                st.success(f"✅ Batting record added! {b_name}: {b_runs} runs ({b_balls} balls)")

    # ─── TAB 3: Add Bowling Data ──────────────────────────────────────────────
    with add_tab3:
        st.markdown("### 🎯 Add Bowling Record")
        with st.form("bowling_form", clear_on_submit=True):
            w_team = st.selectbox("Bowler's Team", TEAMS, key="aw_t")
            bowlers_list = BOWLERS.get(w_team, [])
            w_names = [b["name"] for b in bowlers_list]
            c1, c2 = st.columns(2)
            with c1:
                w_name = st.selectbox("Bowler", w_names, key="aw_n")
                w_opponent = st.selectbox("Opponent", [t for t in TEAMS if t != w_team], key="aw_o")
                w_venue = st.selectbox("Venue", list(VENUES.keys()), key="aw_v")
                w_format = st.selectbox("Format", ["ODI", "T20"], key="aw_f")
            with c2:
                w_overs = st.number_input("Overs Bowled", min_value=1, max_value=10, value=6, key="aw_ov")
                w_runs = st.number_input("Runs Conceded", min_value=0, max_value=150, value=35, key="aw_r")
                w_wickets = st.number_input("Wickets Taken", min_value=0, max_value=10, value=2, key="aw_w")
                w_maidens = st.number_input("Maidens", min_value=0, max_value=10, value=0, key="aw_m")

            if st.form_submit_button("➕ Add Bowling Record", use_container_width=True, type="primary"):
                w_info = next((b for b in bowlers_list if b["name"] == w_name), bowlers_list[0])
                venue_info = VENUES[w_venue]
                country_map = {"India":"India","Australia":"Australia","England":"England",
                    "South Africa":"South Africa","New Zealand":"New Zealand","Pakistan":"Pakistan",
                    "Sri Lanka":"Sri Lanka","Bangladesh":"Bangladesh","West Indies":"West Indies"}
                is_home = 1 if country_map.get(venue_info["country"]) == w_team else 0
                dot_balls = int(w_overs * 6 * 0.4)
                try:
                    existing = pd.read_csv(os.path.join(data_dir, "bowling.csv"))
                    new_id = existing["innings_id"].max() + 1
                except:
                    new_id = 1
                new_row = pd.DataFrame([{
                    "innings_id": new_id, "bowler": w_name, "team": w_team,
                    "opponent": w_opponent, "venue": w_venue,
                    "pitch_type": venue_info["pitch"], "match_format": w_format,
                    "bowling_avg": w_info["avg"], "bowling_sr": w_info["sr"],
                    "bowling_econ": w_info["econ"], "bowling_type": w_info["type"],
                    "is_home": is_home, "opponent_strength": TEAM_STRENGTH[w_opponent],
                    "overs_bowled": w_overs, "runs_conceded": w_runs,
                    "wickets_taken": w_wickets, "maidens": w_maidens,
                    "dot_balls": dot_balls,
                }])
                filepath = os.path.join(data_dir, "bowling.csv")
                new_row.to_csv(filepath, mode="a", header=not os.path.exists(filepath), index=False)
                st.success(f"✅ Bowling record added! {w_name}: {w_wickets}/{w_runs} ({w_overs} overs)")

    # ─── Retrain Models Button ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Retrain Models")
    st.markdown("After adding new data, retrain the models to improve predictions.")
    if st.button("🔄 Retrain All Models", use_container_width=True, type="primary"):
        with st.spinner("Training models... This may take 30-60 seconds."):
            project_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(project_dir, "models")
            matches_df = pd.read_csv(os.path.join(data_dir, "matches.csv"))
            batting_df = pd.read_csv(os.path.join(data_dir, "batting.csv"))
            bowling_df = pd.read_csv(os.path.join(data_dir, "bowling.csv"))
            mp = MatchPredictor()
            mp.train(matches_df)
            mp.save(os.path.join(models_dir, "match_model.pkl"))
            rp = RunsPredictor()
            rp.train(batting_df)
            rp.save(os.path.join(models_dir, "runs_model.pkl"))
            wp = WicketsPredictor()
            wp.train(bowling_df)
            wp.save(os.path.join(models_dir, "wickets_model.pkl"))
            st.success(f"✅ All models retrained!")
            st.metric("Match Accuracy", f"{mp.accuracy*100:.1f}%")
            st.metric("Runs MAE", f"{rp.mae:.1f} runs")
            st.metric("Wickets MAE", f"{wp.mae:.2f}")
            st.info("🔄 Please refresh the page to use the updated models.")

    # ─── Show current data counts ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Current Data Summary")
    try:
        c1, c2, c3 = st.columns(3)
        m_count = len(pd.read_csv(os.path.join(data_dir, "matches.csv")))
        b_count = len(pd.read_csv(os.path.join(data_dir, "batting.csv")))
        w_count = len(pd.read_csv(os.path.join(data_dir, "bowling.csv")))
        c1.metric("🏆 Match Records", f"{m_count:,}")
        c2.metric("🏏 Batting Records", f"{b_count:,}")
        c3.metric("🎯 Bowling Records", f"{w_count:,}")
    except:
        st.warning("No data files found yet.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: IPL Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 IPL Explorer":
    st.markdown("# 🏆 IPL Data Explorer")
    st.markdown("*Explore Indian Premier League match data, team stats & trends*")
    st.markdown("---")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    try:
        ipl_df = pd.read_csv(os.path.join(data_dir, "ipl.csv"))

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏏 Total Matches", f"{len(ipl_df):,}")
        c2.metric("🏟️ Venues", f"{ipl_df['venue'].nunique()}")
        c3.metric("📅 Seasons", f"{ipl_df['season'].nunique()}")
        c4.metric("🏆 Teams", f"{ipl_df['winner'].nunique()}")

        st.markdown("---")

        # Tabs for different views
        ipl_tab1, ipl_tab2, ipl_tab3, ipl_tab4, ipl_tab5 = st.tabs([
            "🏆 Team Wins", "📅 Season Analysis", "🏟️ Venue Stats",
            "🪙 Toss Analysis", "⚔️ Head to Head"
        ])

        # ─── TAB 1: Team Wins ──────────────────────────────────────────
        with ipl_tab1:
            st.markdown("### 🏆 IPL Team Win Count")
            win_counts = ipl_df["winner"].value_counts()
            fig = px.bar(
                x=win_counts.index, y=win_counts.values,
                title="Total Wins by Team",
                labels={"x": "Team", "y": "Wins"},
                color=win_counts.values,
                color_continuous_scale="YlOrRd"
            )
            fig.update_layout(template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # Win percentage
            st.markdown("### 📊 Win Percentage")
            team_matches = {}
            for _, row in ipl_df.iterrows():
                for t in [row["team1"], row["team2"]]:
                    team_matches[t] = team_matches.get(t, 0) + 1
            win_pct = {t: (win_counts.get(t, 0) / team_matches.get(t, 1)) * 100
                       for t in IPL_TEAMS}
            win_pct_sorted = dict(sorted(win_pct.items(), key=lambda x: x[1], reverse=True))
            fig2 = px.bar(
                x=list(win_pct_sorted.keys()), y=list(win_pct_sorted.values()),
                title="Win Percentage",
                labels={"x": "Team", "y": "Win %"},
                color=list(win_pct_sorted.values()),
                color_continuous_scale="Viridis"
            )
            fig2.update_layout(template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

        # ─── TAB 2: Season Analysis ────────────────────────────────────
        with ipl_tab2:
            st.markdown("### 📅 Season-wise Analysis")
            season = st.selectbox("Select Season", sorted(ipl_df["season"].unique(), reverse=True), key="ipl_season")
            season_df = ipl_df[ipl_df["season"] == season]

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Matches Played", len(season_df))
            sc2.metric("Avg Score", f"{(season_df['team1_score'].mean() + season_df['team2_score'].mean()) / 2:.0f}")
            sc3.metric("Highest Score", max(season_df["team1_score"].max(), season_df["team2_score"].max()))

            season_wins = season_df["winner"].value_counts()
            fig = px.pie(
                names=season_wins.index, values=season_wins.values,
                title=f"IPL {season} — Wins by Team",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig, use_container_width=True)

            # Score distribution per season
            all_scores = pd.concat([
                season_df["team1_score"].rename("score"),
                season_df["team2_score"].rename("score")
            ])
            fig3 = px.histogram(
                all_scores, nbins=25,
                title=f"Score Distribution — IPL {season}",
                labels={"value": "Score", "count": "Frequency"},
                color_discrete_sequence=["#FF6B35"]
            )
            fig3.update_layout(template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        # ─── TAB 3: Venue Stats ────────────────────────────────────────
        with ipl_tab3:
            st.markdown("### 🏟️ Venue Statistics")
            venue_stats = ipl_df.groupby("venue").agg(
                matches=("match_id", "count"),
                avg_score_t1=("team1_score", "mean"),
                avg_score_t2=("team2_score", "mean"),
                highest=("team1_score", "max"),
            ).reset_index()
            venue_stats["avg_score"] = (venue_stats["avg_score_t1"] + venue_stats["avg_score_t2"]) / 2
            venue_stats = venue_stats.sort_values("avg_score", ascending=False)

            fig = px.bar(
                venue_stats, x="venue", y="avg_score",
                title="Average Score by Venue",
                labels={"venue": "Venue", "avg_score": "Avg Score"},
                color="avg_score", color_continuous_scale="Magma"
            )
            fig.update_layout(template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(venue_stats[["venue", "matches", "avg_score", "highest"]].round(1),
                         use_container_width=True)

        # ─── TAB 4: Toss Analysis ──────────────────────────────────────
        with ipl_tab4:
            st.markdown("### 🪙 Toss Impact Analysis")

            # Toss winner = match winner?
            ipl_df["toss_winner_won"] = ipl_df["toss_winner"] == ipl_df["winner"]
            toss_impact = ipl_df["toss_winner_won"].value_counts()
            fig = px.pie(
                names=["Toss Winner Won Match", "Toss Winner Lost Match"],
                values=[toss_impact.get(True, 0), toss_impact.get(False, 0)],
                title="Does Winning Toss = Winning Match?",
                color_discrete_sequence=["#F7C948", "#8B5CF6"]
            )
            fig.update_layout(template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig, use_container_width=True)

            # Bat vs Field decision impact
            toss_dec = ipl_df.groupby("toss_decision").apply(
                lambda x: (x["toss_winner"] == x["winner"]).mean() * 100
            ).reset_index()
            toss_dec.columns = ["Decision", "Win %"]
            tc1, tc2 = st.columns(2)
            tc1.metric("🏏 Bat First Win %", f"{toss_dec[toss_dec['Decision']=='bat']['Win %'].values[0]:.1f}%")
            tc2.metric("🏟️ Field First Win %", f"{toss_dec[toss_dec['Decision']=='field']['Win %'].values[0]:.1f}%")

        # ─── TAB 5: Head to Head ───────────────────────────────────────
        with ipl_tab5:
            st.markdown("### ⚔️ Head-to-Head Comparison")
            hc1, hc2 = st.columns(2)
            with hc1:
                h2h_team1 = st.selectbox("Team 1", IPL_TEAMS, index=0, key="h2h_t1")
            with hc2:
                h2h_team2 = st.selectbox("Team 2", [t for t in IPL_TEAMS if t != h2h_team1], key="h2h_t2")

            h2h = ipl_df[((ipl_df["team1"] == h2h_team1) & (ipl_df["team2"] == h2h_team2)) |
                         ((ipl_df["team1"] == h2h_team2) & (ipl_df["team2"] == h2h_team1))]

            if len(h2h) > 0:
                t1_wins = len(h2h[h2h["winner"] == h2h_team1])
                t2_wins = len(h2h[h2h["winner"] == h2h_team2])
                hm1, hm2, hm3 = st.columns(3)
                hm1.metric(f"{IPL_TEAM_SHORT[h2h_team1]} Wins", t1_wins)
                hm2.metric("Total Matches", len(h2h))
                hm3.metric(f"{IPL_TEAM_SHORT[h2h_team2]} Wins", t2_wins)

                fig = go.Figure(go.Bar(
                    x=[IPL_TEAM_SHORT[h2h_team1], IPL_TEAM_SHORT[h2h_team2]],
                    y=[t1_wins, t2_wins],
                    marker=dict(color=["#FF6B35", "#8B5CF6"]),
                    text=[t1_wins, t2_wins], textposition="auto"
                ))
                fig.update_layout(title=f"{IPL_TEAM_SHORT[h2h_team1]} vs {IPL_TEAM_SHORT[h2h_team2]} — Head to Head",
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 📋 Recent Matches")
                display_cols = ["season", "team1_short", "team2_short", "venue",
                                "team1_score", "team2_score", "winner_short",
                                "win_margin", "player_of_match"]
                st.dataframe(h2h[display_cols].sort_values("season", ascending=False).head(20),
                             use_container_width=True)
            else:
                st.info("No head-to-head matches found between these teams.")

        # ─── Full IPL Data Table ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Full IPL Dataset")
        display_cols = ["season", "team1_short", "team2_short", "city",
                        "team1_score", "team2_score", "winner_short",
                        "win_margin", "player_of_match", "match_phase"]
        sorted_df = ipl_df.sort_values("season", ascending=False)
        st.dataframe(sorted_df[display_cols].head(100), use_container_width=True)

    except FileNotFoundError:
        st.warning("IPL data not found. Generating...")
        from data.generate_ipl_data import generate_ipl_data
        ipl_df = generate_ipl_data(2000)
        ipl_df.to_csv(os.path.join(data_dir, "ipl.csv"), index=False)
        st.success("✅ IPL data generated! Please refresh the page.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: IPL Match Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏏 IPL Prediction":
    st.markdown("# 🏏 IPL Match Prediction")
    st.markdown("*Predict which IPL team will win based on historical data*")
    st.markdown("---")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    try:
        ipl_df = pd.read_csv(os.path.join(data_dir, "ipl.csv"))

        col1, col2 = st.columns(2)
        with col1:
            ipl_t1 = st.selectbox("🏏 Team 1", IPL_TEAMS, index=0, key="ipl_pt1")
        with col2:
            ipl_t2 = st.selectbox("🏏 Team 2", [t for t in IPL_TEAMS if t != ipl_t1], key="ipl_pt2")

        col3, col4 = st.columns(2)
        with col3:
            ipl_venue = st.selectbox("🏟️ Venue", list(IPL_VENUES.keys()), key="ipl_pv")
        with col4:
            ipl_toss_winner = st.selectbox("🪙 Toss Winner", [ipl_t1, ipl_t2], key="ipl_ptw")
        ipl_toss_decision = st.selectbox("🪙 Toss Decision", ["bat", "field"], key="ipl_ptd")

        if st.button("🔮 Predict IPL Winner", use_container_width=True, type="primary"):
            # Calculate prediction based on historical data
            h2h = ipl_df[((ipl_df["team1"] == ipl_t1) & (ipl_df["team2"] == ipl_t2)) |
                         ((ipl_df["team1"] == ipl_t2) & (ipl_df["team2"] == ipl_t1))]

            # Base probability from team strength
            s1 = IPL_TEAM_STRENGTH[ipl_t1]
            s2 = IPL_TEAM_STRENGTH[ipl_t2]
            strength_prob = s1 / (s1 + s2)

            # H2H adjustment
            if len(h2h) > 0:
                t1_h2h_wins = len(h2h[h2h["winner"] == ipl_t1])
                h2h_ratio = t1_h2h_wins / len(h2h)
                strength_prob = 0.6 * strength_prob + 0.4 * h2h_ratio

            # Home advantage
            venue_info = IPL_VENUES[ipl_venue]
            if venue_info["home_team"] == ipl_t1:
                strength_prob += 0.05
            elif venue_info["home_team"] == ipl_t2:
                strength_prob -= 0.05

            # Toss advantage
            if ipl_toss_winner == ipl_t1:
                strength_prob += 0.02
            else:
                strength_prob -= 0.02

            strength_prob = max(0.1, min(0.9, strength_prob))
            t1_prob = round(strength_prob * 100, 1)
            t2_prob = round((1 - strength_prob) * 100, 1)
            winner = ipl_t1 if strength_prob > 0.5 else ipl_t2

            st.markdown("---")
            st.markdown(f'<div class="prediction-card"><div class="winner-badge glow">🏆 {winner} WINS!</div></div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric(f"🏏 {IPL_TEAM_SHORT[ipl_t1]}", f"{t1_prob}%")
            c2.metric("🏟️ Venue", venue_info["city"])
            c3.metric(f"🏏 {IPL_TEAM_SHORT[ipl_t2]}", f"{t2_prob}%")

            fig = go.Figure(go.Bar(
                x=[IPL_TEAM_SHORT[ipl_t1], IPL_TEAM_SHORT[ipl_t2]],
                y=[t1_prob, t2_prob],
                marker=dict(color=["#FF6B35", "#8B5CF6"]),
                text=[f"{t1_prob}%", f"{t2_prob}%"], textposition="auto"
            ))
            fig.update_layout(title="Win Probability", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), yaxis_title="Probability (%)")
            st.plotly_chart(fig, use_container_width=True)

            # Show H2H history
            if len(h2h) > 0:
                st.markdown(f"### 📊 {IPL_TEAM_SHORT[ipl_t1]} vs {IPL_TEAM_SHORT[ipl_t2]} — Past Matches")
                t1_wins = len(h2h[h2h["winner"] == ipl_t1])
                t2_wins = len(h2h[h2h["winner"] == ipl_t2])
                hc1, hc2, hc3 = st.columns(3)
                hc1.metric(f"{IPL_TEAM_SHORT[ipl_t1]} Wins", t1_wins)
                hc2.metric("Total Matches", len(h2h))
                hc3.metric(f"{IPL_TEAM_SHORT[ipl_t2]} Wins", t2_wins)

    except FileNotFoundError:
        st.warning("IPL data not found. Please go to IPL Explorer first.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: IPL Teams
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 IPL Teams":
    st.markdown("# 👥 IPL Teams & Players")
    st.markdown("*Browse all IPL teams, their players, and stats*")
    st.markdown("---")

    selected_team = st.selectbox("Select Team", IPL_TEAMS, key="ipl_team_select")

    # Team header
    st.markdown(f"## {selected_team} ({IPL_TEAM_SHORT[selected_team]})")
    st.metric("Team Strength Rating", f"{IPL_TEAM_STRENGTH[selected_team]}/100")

    # Get home venue
    home_venue = [v for v, info in IPL_VENUES.items() if info["home_team"] == selected_team]
    if home_venue:
        venue_info = IPL_VENUES[home_venue[0]]
        vc1, vc2 = st.columns(2)
        vc1.metric("🏟️ Home Ground", home_venue[0].split(",")[0])
        vc2.metric("📍 City / Pitch", f"{venue_info['city']} ({venue_info['pitch']})")

    team_data = IPL_PLAYERS.get(selected_team, {})

    # Batsmen
    st.markdown("---")
    st.markdown("### 🏏 Batsmen")
    batsmen = team_data.get("batsmen", [])
    if batsmen:
        bat_df = pd.DataFrame(batsmen)
        bat_df.columns = ["Player", "Average", "Strike Rate", "Role"]
        bat_df.index = range(1, len(bat_df) + 1)

        # Cards view
        cols = st.columns(min(len(batsmen), 5))
        for idx, b in enumerate(batsmen):
            with cols[idx % len(cols)]:
                st.markdown(f"""
                <div style="background:rgba(255,107,53,0.15); border-radius:16px;
                     padding:16px; margin:6px 0; border:1px solid rgba(255,107,53,0.3);">
                    <h4 style="color:#F7C948; margin:0;">{b['name']}</h4>
                    <p style="color:#94a3b8; margin:4px 0;">{b['role'].title()}</p>
                    <p style="color:#e2e8f0;">Avg: <b>{b['avg']}</b> | SR: <b>{b['sr']}</b></p>
                </div>
                """, unsafe_allow_html=True)

        # Stats chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Average", x=[b["name"] for b in batsmen],
            y=[b["avg"] for b in batsmen], marker_color="#FF6B35"))
        fig.add_trace(go.Bar(name="Strike Rate", x=[b["name"] for b in batsmen],
            y=[b["sr"] for b in batsmen], marker_color="#8B5CF6"))
        fig.update_layout(title=f"{IPL_TEAM_SHORT[selected_team]} Batsmen Stats",
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True)

    # Bowlers
    st.markdown("---")
    st.markdown("### 🎯 Bowlers")
    bowlers = team_data.get("bowlers", [])
    if bowlers:
        cols = st.columns(min(len(bowlers), 5))
        for idx, b in enumerate(bowlers):
            with cols[idx % len(cols)]:
                st.markdown(f"""
                <div style="background:rgba(139,92,246,0.15); border-radius:16px;
                     padding:16px; margin:6px 0; border:1px solid rgba(139,92,246,0.3);">
                    <h4 style="color:#F7C948; margin:0;">{b['name']}</h4>
                    <p style="color:#94a3b8; margin:4px 0;">{b['type'].title()}</p>
                    <p style="color:#e2e8f0;">Avg: <b>{b['avg']}</b> | Econ: <b>{b['econ']}</b> | SR: <b>{b['sr']}</b></p>
                </div>
                """, unsafe_allow_html=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Bowling Avg", x=[b["name"] for b in bowlers],
            y=[b["avg"] for b in bowlers], marker_color="#F7C948"))
        fig2.add_trace(go.Bar(name="Economy", x=[b["name"] for b in bowlers],
            y=[b["econ"] for b in bowlers], marker_color="#3B82F6"))
        fig2.update_layout(title=f"{IPL_TEAM_SHORT[selected_team]} Bowlers Stats",
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"))
        st.plotly_chart(fig2, use_container_width=True)

    # Team record from IPL data
    st.markdown("---")
    st.markdown("### 📊 Team Record")
    try:
        ipl_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ipl.csv"))
        team_matches = ipl_df[(ipl_df["team1"] == selected_team) | (ipl_df["team2"] == selected_team)]
        total = len(team_matches)
        wins = len(team_matches[team_matches["winner"] == selected_team])
        losses = total - wins
        win_pct = (wins / total * 100) if total > 0 else 0

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Matches", total)
        rc2.metric("Wins", wins)
        rc3.metric("Losses", losses)
        rc4.metric("Win %", f"{win_pct:.1f}%")

        # Season-wise performance
        season_perf = []
        for season in sorted(ipl_df["season"].unique()):
            s_df = ipl_df[(ipl_df["season"] == season) &
                          ((ipl_df["team1"] == selected_team) | (ipl_df["team2"] == selected_team))]
            s_wins = len(s_df[s_df["winner"] == selected_team])
            season_perf.append({"Season": season, "Matches": len(s_df), "Wins": s_wins})
        perf_df = pd.DataFrame(season_perf)
        if len(perf_df) > 0:
            fig3 = px.bar(perf_df, x="Season", y="Wins",
                title=f"{IPL_TEAM_SHORT[selected_team]} — Wins per Season",
                color="Wins", color_continuous_scale="YlOrRd")
            fig3.update_layout(template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"))
            st.plotly_chart(fig3, use_container_width=True)
    except:
        pass
