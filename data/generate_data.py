"""
Cricket Data Generator
=====================
Generates realistic synthetic cricket match, batting, and bowling data
for training ML prediction models.
"""

import pandas as pd
import numpy as np
import os

# ─── Constants ───────────────────────────────────────────────────────────────

TEAMS = [
    "India", "Australia", "England", "South Africa", "New Zealand",
    "Pakistan", "Sri Lanka", "Bangladesh", "West Indies", "Afghanistan"
]

VENUES = {
    "Wankhede Stadium, Mumbai": {"country": "India", "pitch": "batting"},
    "Eden Gardens, Kolkata": {"country": "India", "pitch": "balanced"},
    "M. Chinnaswamy Stadium, Bangalore": {"country": "India", "pitch": "batting"},
    "MA Chidambaram Stadium, Chennai": {"country": "India", "pitch": "spin"},
    "Melbourne Cricket Ground": {"country": "Australia", "pitch": "pace"},
    "Sydney Cricket Ground": {"country": "Australia", "pitch": "balanced"},
    "Lord's, London": {"country": "England", "pitch": "pace"},
    "The Oval, London": {"country": "England", "pitch": "balanced"},
    "Newlands, Cape Town": {"country": "South Africa", "pitch": "pace"},
    "Gaddafi Stadium, Lahore": {"country": "Pakistan", "pitch": "spin"},
    "Dubai International Stadium": {"country": "UAE", "pitch": "spin"},
    "Hagley Oval, Christchurch": {"country": "New Zealand", "pitch": "pace"},
    "Kensington Oval, Barbados": {"country": "West Indies", "pitch": "batting"},
    "R. Premadasa Stadium, Colombo": {"country": "Sri Lanka", "pitch": "spin"},
    "Shere Bangla Stadium, Dhaka": {"country": "Bangladesh", "pitch": "spin"},
}

BATSMEN = {
    "India": [
        {"name": "Virat Kohli", "avg": 59.1, "sr": 93.2, "style": "right", "role": "top"},
        {"name": "Rohit Sharma", "avg": 49.7, "sr": 91.5, "style": "right", "role": "opener"},
        {"name": "Shubman Gill", "avg": 52.3, "sr": 98.1, "style": "right", "role": "opener"},
        {"name": "KL Rahul", "avg": 47.5, "sr": 87.3, "style": "right", "role": "top"},
        {"name": "Shreyas Iyer", "avg": 44.2, "sr": 96.7, "style": "right", "role": "middle"},
    ],
    "Australia": [
        {"name": "Steve Smith", "avg": 56.8, "sr": 86.4, "style": "right", "role": "top"},
        {"name": "David Warner", "avg": 45.3, "sr": 97.1, "style": "left", "role": "opener"},
        {"name": "Marnus Labuschagne", "avg": 53.1, "sr": 84.2, "style": "right", "role": "top"},
        {"name": "Travis Head", "avg": 44.9, "sr": 101.3, "style": "left", "role": "middle"},
        {"name": "Mitchell Marsh", "avg": 38.5, "sr": 92.6, "style": "right", "role": "middle"},
    ],
    "England": [
        {"name": "Joe Root", "avg": 51.2, "sr": 87.5, "style": "right", "role": "top"},
        {"name": "Ben Stokes", "avg": 40.1, "sr": 95.8, "style": "left", "role": "middle"},
        {"name": "Harry Brook", "avg": 48.7, "sr": 103.2, "style": "right", "role": "middle"},
        {"name": "Jonny Bairstow", "avg": 42.3, "sr": 104.5, "style": "right", "role": "middle"},
        {"name": "Jos Buttler", "avg": 39.8, "sr": 118.2, "style": "right", "role": "lower"},
    ],
    "Pakistan": [
        {"name": "Babar Azam", "avg": 56.2, "sr": 88.9, "style": "right", "role": "top"},
        {"name": "Fakhar Zaman", "avg": 42.1, "sr": 94.3, "style": "left", "role": "opener"},
        {"name": "Mohammad Rizwan", "avg": 44.8, "sr": 81.2, "style": "right", "role": "middle"},
        {"name": "Imam-ul-Haq", "avg": 49.5, "sr": 79.1, "style": "left", "role": "opener"},
        {"name": "Saud Shakeel", "avg": 46.3, "sr": 82.4, "style": "left", "role": "middle"},
    ],
    "South Africa": [
        {"name": "Quinton de Kock", "avg": 44.6, "sr": 96.3, "style": "left", "role": "opener"},
        {"name": "Aiden Markram", "avg": 38.9, "sr": 87.1, "style": "right", "role": "top"},
        {"name": "Rassie van der Dussen", "avg": 48.2, "sr": 84.5, "style": "right", "role": "middle"},
        {"name": "Heinrich Klaasen", "avg": 41.5, "sr": 112.7, "style": "right", "role": "middle"},
        {"name": "David Miller", "avg": 36.8, "sr": 101.4, "style": "left", "role": "lower"},
    ],
    "New Zealand": [
        {"name": "Kane Williamson", "avg": 48.5, "sr": 81.3, "style": "right", "role": "top"},
        {"name": "Devon Conway", "avg": 45.2, "sr": 85.7, "style": "left", "role": "opener"},
        {"name": "Tom Latham", "avg": 39.8, "sr": 78.4, "style": "left", "role": "opener"},
        {"name": "Glenn Phillips", "avg": 35.6, "sr": 98.2, "style": "right", "role": "middle"},
        {"name": "Daryl Mitchell", "avg": 43.1, "sr": 93.5, "style": "right", "role": "middle"},
    ],
    "Sri Lanka": [
        {"name": "Pathum Nissanka", "avg": 39.4, "sr": 85.6, "style": "right", "role": "opener"},
        {"name": "Kusal Mendis", "avg": 33.8, "sr": 89.2, "style": "right", "role": "top"},
        {"name": "Charith Asalanka", "avg": 35.2, "sr": 92.1, "style": "left", "role": "middle"},
        {"name": "Sadeera Samarawickrama", "avg": 31.5, "sr": 84.3, "style": "right", "role": "middle"},
        {"name": "Dhananjaya de Silva", "avg": 37.1, "sr": 86.5, "style": "right", "role": "middle"},
    ],
    "Bangladesh": [
        {"name": "Shakib Al Hasan", "avg": 37.8, "sr": 82.1, "style": "left", "role": "middle"},
        {"name": "Mushfiqur Rahim", "avg": 36.5, "sr": 78.9, "style": "right", "role": "middle"},
        {"name": "Litton Das", "avg": 34.2, "sr": 88.4, "style": "right", "role": "opener"},
        {"name": "Tamim Iqbal", "avg": 36.1, "sr": 80.5, "style": "left", "role": "opener"},
        {"name": "Mahmudullah", "avg": 33.9, "sr": 76.8, "style": "right", "role": "lower"},
    ],
    "West Indies": [
        {"name": "Shai Hope", "avg": 40.2, "sr": 78.6, "style": "right", "role": "opener"},
        {"name": "Nicholas Pooran", "avg": 30.5, "sr": 105.3, "style": "left", "role": "middle"},
        {"name": "Brandon King", "avg": 35.8, "sr": 91.2, "style": "right", "role": "opener"},
        {"name": "Shimron Hetmyer", "avg": 31.2, "sr": 98.7, "style": "left", "role": "middle"},
        {"name": "Kyle Mayers", "avg": 33.5, "sr": 95.1, "style": "left", "role": "middle"},
    ],
    "Afghanistan": [
        {"name": "Rahmanullah Gurbaz", "avg": 38.1, "sr": 102.5, "style": "right", "role": "opener"},
        {"name": "Ibrahim Zadran", "avg": 41.2, "sr": 82.3, "style": "right", "role": "opener"},
        {"name": "Hashmatullah Shahidi", "avg": 35.6, "sr": 72.4, "style": "left", "role": "top"},
        {"name": "Najibullah Zadran", "avg": 28.9, "sr": 108.7, "style": "left", "role": "middle"},
        {"name": "Azmatullah Omarzai", "avg": 32.4, "sr": 96.8, "style": "right", "role": "middle"},
    ],
}

BOWLERS = {
    "India": [
        {"name": "Jasprit Bumrah", "avg": 24.3, "sr": 30.2, "econ": 4.5, "type": "fast"},
        {"name": "Mohammed Shami", "avg": 26.1, "sr": 28.7, "econ": 5.2, "type": "fast"},
        {"name": "Ravindra Jadeja", "avg": 32.5, "sr": 42.1, "econ": 4.8, "type": "spin"},
        {"name": "Kuldeep Yadav", "avg": 27.8, "sr": 33.5, "econ": 5.1, "type": "spin"},
        {"name": "Mohammed Siraj", "avg": 28.4, "sr": 31.9, "econ": 5.4, "type": "fast"},
    ],
    "Australia": [
        {"name": "Pat Cummins", "avg": 25.7, "sr": 29.8, "econ": 5.0, "type": "fast"},
        {"name": "Mitchell Starc", "avg": 24.8, "sr": 27.3, "econ": 5.3, "type": "fast"},
        {"name": "Josh Hazlewood", "avg": 25.1, "sr": 30.5, "econ": 4.7, "type": "fast"},
        {"name": "Adam Zampa", "avg": 29.3, "sr": 35.2, "econ": 5.1, "type": "spin"},
        {"name": "Glenn Maxwell", "avg": 42.1, "sr": 52.3, "econ": 5.5, "type": "spin"},
    ],
    "England": [
        {"name": "Mark Wood", "avg": 28.9, "sr": 31.2, "econ": 5.6, "type": "fast"},
        {"name": "Chris Woakes", "avg": 30.2, "sr": 35.1, "econ": 5.1, "type": "fast"},
        {"name": "Adil Rashid", "avg": 33.5, "sr": 38.7, "econ": 5.4, "type": "spin"},
        {"name": "Reece Topley", "avg": 26.7, "sr": 29.8, "econ": 5.2, "type": "fast"},
        {"name": "Moeen Ali", "avg": 36.8, "sr": 43.2, "econ": 5.3, "type": "spin"},
    ],
    "Pakistan": [
        {"name": "Shaheen Afridi", "avg": 24.1, "sr": 27.5, "econ": 5.1, "type": "fast"},
        {"name": "Haris Rauf", "avg": 27.3, "sr": 30.1, "econ": 5.6, "type": "fast"},
        {"name": "Shadab Khan", "avg": 30.8, "sr": 36.4, "econ": 5.2, "type": "spin"},
        {"name": "Naseem Shah", "avg": 28.5, "sr": 32.1, "econ": 5.3, "type": "fast"},
        {"name": "Mohammad Nawaz", "avg": 35.2, "sr": 41.5, "econ": 5.0, "type": "spin"},
    ],
    "South Africa": [
        {"name": "Kagiso Rabada", "avg": 25.4, "sr": 28.6, "econ": 5.2, "type": "fast"},
        {"name": "Anrich Nortje", "avg": 24.8, "sr": 26.9, "econ": 5.5, "type": "fast"},
        {"name": "Marco Jansen", "avg": 27.1, "sr": 31.4, "econ": 5.0, "type": "fast"},
        {"name": "Keshav Maharaj", "avg": 31.5, "sr": 38.2, "econ": 4.8, "type": "spin"},
        {"name": "Lungi Ngidi", "avg": 26.3, "sr": 29.7, "econ": 5.4, "type": "fast"},
    ],
    "New Zealand": [
        {"name": "Trent Boult", "avg": 25.6, "sr": 29.1, "econ": 5.1, "type": "fast"},
        {"name": "Tim Southee", "avg": 29.8, "sr": 33.5, "econ": 5.4, "type": "fast"},
        {"name": "Matt Henry", "avg": 26.2, "sr": 28.8, "econ": 5.0, "type": "fast"},
        {"name": "Mitchell Santner", "avg": 35.1, "sr": 42.3, "econ": 4.7, "type": "spin"},
        {"name": "Lockie Ferguson", "avg": 24.5, "sr": 26.4, "econ": 5.3, "type": "fast"},
    ],
    "Sri Lanka": [
        {"name": "Wanindu Hasaranga", "avg": 24.8, "sr": 28.2, "econ": 5.0, "type": "spin"},
        {"name": "Maheesh Theekshana", "avg": 26.5, "sr": 30.1, "econ": 4.8, "type": "spin"},
        {"name": "Dilshan Madushanka", "avg": 28.3, "sr": 32.5, "econ": 5.3, "type": "fast"},
        {"name": "Dushmantha Chameera", "avg": 29.1, "sr": 33.8, "econ": 5.5, "type": "fast"},
        {"name": "Dunith Wellalage", "avg": 30.2, "sr": 35.4, "econ": 4.9, "type": "spin"},
    ],
    "Bangladesh": [
        {"name": "Mustafizur Rahman", "avg": 28.7, "sr": 33.1, "econ": 5.0, "type": "fast"},
        {"name": "Taskin Ahmed", "avg": 30.5, "sr": 35.8, "econ": 5.4, "type": "fast"},
        {"name": "Mehidy Hasan Miraz", "avg": 32.1, "sr": 38.5, "econ": 4.7, "type": "spin"},
        {"name": "Shakib Al Hasan", "avg": 33.8, "sr": 40.2, "econ": 4.6, "type": "spin"},
        {"name": "Shoriful Islam", "avg": 31.4, "sr": 36.2, "econ": 5.5, "type": "fast"},
    ],
    "West Indies": [
        {"name": "Alzarri Joseph", "avg": 27.5, "sr": 30.8, "econ": 5.3, "type": "fast"},
        {"name": "Jason Holder", "avg": 31.2, "sr": 37.5, "econ": 5.0, "type": "fast"},
        {"name": "Akeal Hosein", "avg": 29.8, "sr": 34.1, "econ": 4.8, "type": "spin"},
        {"name": "Gudakesh Motie", "avg": 28.5, "sr": 32.6, "econ": 4.5, "type": "spin"},
        {"name": "Jayden Seales", "avg": 30.1, "sr": 35.2, "econ": 5.4, "type": "fast"},
    ],
    "Afghanistan": [
        {"name": "Rashid Khan", "avg": 18.5, "sr": 22.3, "econ": 4.1, "type": "spin"},
        {"name": "Mujeeb Ur Rahman", "avg": 25.1, "sr": 29.5, "econ": 4.5, "type": "spin"},
        {"name": "Fazalhaq Farooqi", "avg": 24.2, "sr": 26.8, "econ": 5.1, "type": "fast"},
        {"name": "Naveen-ul-Haq", "avg": 26.8, "sr": 30.2, "econ": 5.3, "type": "fast"},
        {"name": "Noor Ahmad", "avg": 27.5, "sr": 31.8, "econ": 4.7, "type": "spin"},
    ],
}

# Team strength ratings (0-100)
TEAM_STRENGTH = {
    "India": 92, "Australia": 90, "England": 85, "South Africa": 84,
    "New Zealand": 83, "Pakistan": 80, "Sri Lanka": 72, "Bangladesh": 65,
    "West Indies": 68, "Afghanistan": 70,
}


def generate_match_data(n_matches=3000):
    """Generate realistic cricket match data."""
    np.random.seed(42)
    records = []

    for i in range(n_matches):
        # Pick two different teams
        team1, team2 = np.random.choice(TEAMS, size=2, replace=False)

        # Pick venue
        venue = np.random.choice(list(VENUES.keys()))
        venue_info = VENUES[venue]
        pitch_type = venue_info["pitch"]
        venue_country = venue_info["country"]

        # Home advantage
        team1_home = 1 if any(team1.lower() in venue_country.lower() for _ in [1]) else 0
        team2_home = 1 if any(team2.lower() in venue_country.lower() for _ in [1]) else 0

        # Determine home advantage more accurately
        country_team_map = {
            "India": "India", "Australia": "Australia", "England": "England",
            "South Africa": "South Africa", "New Zealand": "New Zealand",
            "Pakistan": "Pakistan", "Sri Lanka": "Sri Lanka",
            "Bangladesh": "Bangladesh", "West Indies": "West Indies",
            "UAE": "Pakistan",  # Pakistan often plays in UAE
        }
        home_team = country_team_map.get(venue_country, "")
        team1_home = 1 if team1 == home_team else 0
        team2_home = 1 if team2 == home_team else 0

        # Toss
        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(["bat", "field"], p=[0.4, 0.6])

        # Match format
        match_format = np.random.choice(["ODI", "T20"], p=[0.5, 0.5])

        # Calculate win probability based on multiple factors
        strength_diff = TEAM_STRENGTH[team1] - TEAM_STRENGTH[team2]
        home_advantage = (team1_home - team2_home) * 8
        toss_advantage = 3 if toss_winner == team1 else -3

        # Pitch factor
        pitch_factor = 0
        if pitch_type == "spin":
            spin_teams = ["India", "Sri Lanka", "Bangladesh", "Afghanistan"]
            if team1 in spin_teams:
                pitch_factor += 5
            if team2 in spin_teams:
                pitch_factor -= 5
        elif pitch_type == "pace":
            pace_teams = ["Australia", "England", "South Africa", "New Zealand"]
            if team1 in pace_teams:
                pitch_factor += 5
            if team2 in pace_teams:
                pitch_factor -= 5

        # Calculate probability
        logit = (strength_diff + home_advantage + toss_advantage + pitch_factor) / 30
        prob_team1_wins = 1 / (1 + np.exp(-logit))
        prob_team1_wins += np.random.normal(0, 0.05)  # Add noise
        prob_team1_wins = np.clip(prob_team1_wins, 0.1, 0.9)

        winner = team1 if np.random.random() < prob_team1_wins else team2

        # Generate scores
        if match_format == "ODI":
            base_score = np.random.normal(260, 40)
            if pitch_type == "batting":
                base_score += 25
            elif pitch_type == "pace":
                base_score -= 15

            team1_score = int(max(100, base_score + np.random.normal(0, 30)))
            if winner == team1:
                team2_score = int(max(80, team1_score - np.random.randint(5, 80)))
            else:
                team2_score = int(max(team1_score + 1, team1_score + np.random.randint(1, 50)))
            max_overs = 50
        else:  # T20
            base_score = np.random.normal(165, 25)
            if pitch_type == "batting":
                base_score += 15
            elif pitch_type == "pace":
                base_score -= 10

            team1_score = int(max(90, base_score + np.random.normal(0, 20)))
            if winner == team1:
                team2_score = int(max(60, team1_score - np.random.randint(3, 50)))
            else:
                team2_score = int(max(team1_score + 1, team1_score + np.random.randint(1, 30)))
            max_overs = 20

        records.append({
            "match_id": i + 1,
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "pitch_type": pitch_type,
            "match_format": match_format,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "team1_home": team1_home,
            "team2_home": team2_home,
            "team1_strength": TEAM_STRENGTH[team1],
            "team2_strength": TEAM_STRENGTH[team2],
            "team1_score": team1_score,
            "team2_score": team2_score,
            "winner": winner,
        })

    return pd.DataFrame(records)


def generate_batting_data(n_innings=5000):
    """Generate realistic batting performance data."""
    np.random.seed(123)
    records = []

    all_batsmen = []
    for team, players in BATSMEN.items():
        for p in players:
            all_batsmen.append({**p, "team": team})

    for i in range(n_innings):
        batsman = np.random.choice(all_batsmen)
        opponent = np.random.choice([t for t in TEAMS if t != batsman["team"]])
        venue = np.random.choice(list(VENUES.keys()))
        venue_info = VENUES[venue]
        match_format = np.random.choice(["ODI", "T20"], p=[0.5, 0.5])

        # Base runs from player average
        base_avg = batsman["avg"]

        # Venue/pitch adjustment
        pitch_bonus = 0
        if venue_info["pitch"] == "batting":
            pitch_bonus = 8
        elif venue_info["pitch"] == "pace":
            pitch_bonus = -5
        elif venue_info["pitch"] == "spin":
            if batsman["style"] == "left":
                pitch_bonus = -3  # Lefties sometimes struggle against spin
            else:
                pitch_bonus = -2

        # Home advantage
        country_team_map = {
            "India": "India", "Australia": "Australia", "England": "England",
            "South Africa": "South Africa", "New Zealand": "New Zealand",
            "Pakistan": "Pakistan", "Sri Lanka": "Sri Lanka",
            "Bangladesh": "Bangladesh", "West Indies": "West Indies",
        }
        is_home = 1 if country_team_map.get(venue_info["country"]) == batsman["team"] else 0
        home_bonus = 7 if is_home else 0

        # Opponent strength adjustment
        opp_bowling_factor = (100 - TEAM_STRENGTH[opponent]) / 20

        # Format adjustment
        if match_format == "T20":
            format_multiplier = 0.65  # Lower scores in T20
        else:
            format_multiplier = 1.0

        # Calculate expected runs with randomness
        expected_runs = (base_avg + pitch_bonus + home_bonus + opp_bowling_factor) * format_multiplier

        # Add significant randomness (cricket is unpredictable!)
        runs = int(max(0, np.random.exponential(expected_runs * 0.7)))

        # Cap runs realistically
        if match_format == "T20":
            runs = min(runs, 120)
        else:
            runs = min(runs, 200)

        # Duck probability (getting out for 0)
        if np.random.random() < 0.08:
            runs = 0

        # Calculate balls faced
        sr = batsman["sr"] + np.random.normal(0, 15)
        sr = max(40, sr)
        balls_faced = max(1, int(runs / (sr / 100)))

        # Calculate 4s and 6s
        fours = int(runs * np.random.uniform(0.25, 0.45) / 4)
        sixes = int(runs * np.random.uniform(0.05, 0.2) / 6)

        # Not out probability
        not_out = 1 if np.random.random() < 0.15 else 0

        # Batting position
        position_map = {"opener": np.random.choice([1, 2]), "top": np.random.choice([3, 4]),
                        "middle": np.random.choice([5, 6]), "lower": np.random.choice([6, 7])}
        batting_position = position_map.get(batsman["role"], 5)

        records.append({
            "innings_id": i + 1,
            "batsman": batsman["name"],
            "team": batsman["team"],
            "opponent": opponent,
            "venue": venue,
            "pitch_type": venue_info["pitch"],
            "match_format": match_format,
            "batting_avg": batsman["avg"],
            "strike_rate": batsman["sr"],
            "batting_style": batsman["style"],
            "batting_position": batting_position,
            "is_home": is_home,
            "opponent_strength": TEAM_STRENGTH[opponent],
            "runs_scored": runs,
            "balls_faced": balls_faced,
            "fours": fours,
            "sixes": sixes,
            "not_out": not_out,
        })

    return pd.DataFrame(records)


def generate_bowling_data(n_innings=5000):
    """Generate realistic bowling performance data."""
    np.random.seed(456)
    records = []

    all_bowlers = []
    for team, players in BOWLERS.items():
        for p in players:
            all_bowlers.append({**p, "team": team})

    for i in range(n_innings):
        bowler = np.random.choice(all_bowlers)
        opponent = np.random.choice([t for t in TEAMS if t != bowler["team"]])
        venue = np.random.choice(list(VENUES.keys()))
        venue_info = VENUES[venue]
        match_format = np.random.choice(["ODI", "T20"], p=[0.5, 0.5])

        # Base wickets from bowling average
        base_wicket_prob = 10 / bowler["avg"]  # Overs per wicket inverse

        # Pitch adjustment
        pitch_bonus = 0
        if venue_info["pitch"] == "pace" and bowler["type"] == "fast":
            pitch_bonus = 0.15
        elif venue_info["pitch"] == "spin" and bowler["type"] == "spin":
            pitch_bonus = 0.2
        elif venue_info["pitch"] == "batting":
            pitch_bonus = -0.1

        # Home advantage
        country_team_map = {
            "India": "India", "Australia": "Australia", "England": "England",
            "South Africa": "South Africa", "New Zealand": "New Zealand",
            "Pakistan": "Pakistan", "Sri Lanka": "Sri Lanka",
            "Bangladesh": "Bangladesh", "West Indies": "West Indies",
        }
        is_home = 1 if country_team_map.get(venue_info["country"]) == bowler["team"] else 0
        home_bonus = 0.1 if is_home else 0

        # Opponent batting weakness
        opp_batting_factor = (100 - TEAM_STRENGTH[opponent]) / 200

        # Overs bowled
        if match_format == "T20":
            overs = np.random.choice([2, 3, 4], p=[0.15, 0.25, 0.6])
            max_wickets = 4
        else:
            overs = np.random.choice([5, 6, 7, 8, 9, 10], p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.3])
            max_wickets = 7

        # Calculate wickets
        wicket_prob_per_over = base_wicket_prob + pitch_bonus + home_bonus + opp_batting_factor
        wickets = 0
        for _ in range(overs):
            if np.random.random() < wicket_prob_per_over:
                wickets += 1
        wickets = min(wickets, max_wickets)

        # Runs conceded
        base_econ = bowler["econ"] + np.random.normal(0, 1.2)
        if venue_info["pitch"] == "batting":
            base_econ += 0.8
        elif venue_info["pitch"] == "pace" and bowler["type"] == "fast":
            base_econ -= 0.5
        elif venue_info["pitch"] == "spin" and bowler["type"] == "spin":
            base_econ -= 0.6

        if match_format == "T20":
            base_econ += 1.5  # T20s are more expensive

        base_econ = max(2.0, base_econ)
        runs_conceded = int(max(0, overs * base_econ + np.random.normal(0, 5)))

        # Maiden overs (only in ODI)
        maidens = 0
        if match_format == "ODI":
            for _ in range(overs):
                if np.random.random() < (0.15 if bowler["econ"] < 5.0 else 0.08):
                    maidens += 1

        # Dot balls
        dot_ball_pct = max(0.2, 0.5 - (base_econ - 4) * 0.05 + np.random.normal(0, 0.05))
        dot_balls = int(overs * 6 * dot_ball_pct)

        records.append({
            "innings_id": i + 1,
            "bowler": bowler["name"],
            "team": bowler["team"],
            "opponent": opponent,
            "venue": venue,
            "pitch_type": venue_info["pitch"],
            "match_format": match_format,
            "bowling_avg": bowler["avg"],
            "bowling_sr": bowler["sr"],
            "bowling_econ": bowler["econ"],
            "bowling_type": bowler["type"],
            "is_home": is_home,
            "opponent_strength": TEAM_STRENGTH[opponent],
            "overs_bowled": overs,
            "runs_conceded": runs_conceded,
            "wickets_taken": wickets,
            "maidens": maidens,
            "dot_balls": dot_balls,
        })

    return pd.DataFrame(records)


def main():
    """Generate and save all datasets."""
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("🏏 Generating Cricket Match Data...")
    match_df = generate_match_data(3000)
    match_df.to_csv(os.path.join(data_dir, "matches.csv"), index=False)
    print(f"   ✅ Generated {len(match_df)} match records")

    print("🏏 Generating Batting Performance Data...")
    batting_df = generate_batting_data(5000)
    batting_df.to_csv(os.path.join(data_dir, "batting.csv"), index=False)
    print(f"   ✅ Generated {len(batting_df)} batting records")

    print("🏏 Generating Bowling Performance Data...")
    bowling_df = generate_bowling_data(5000)
    bowling_df.to_csv(os.path.join(data_dir, "bowling.csv"), index=False)
    print(f"   ✅ Generated {len(bowling_df)} bowling records")

    print("\n📊 Data Summary:")
    print(f"   Matches: {match_df.shape}")
    print(f"   Batting: {batting_df.shape}")
    print(f"   Bowling: {bowling_df.shape}")
    print("\n✅ All data saved to 'data/' directory!")


if __name__ == "__main__":
    main()
