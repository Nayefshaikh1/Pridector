"""
IPL Data Generator
==================
Generates realistic synthetic IPL (Indian Premier League) match data
with all current IPL teams, players, and venues.
"""

import pandas as pd
import numpy as np
import os

# ─── IPL Teams ────────────────────────────────────────────────────────────────
IPL_TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
    "Sunrisers Hyderabad", "Punjab Kings", "Lucknow Super Giants",
    "Gujarat Titans"
]

IPL_TEAM_SHORT = {
    "Mumbai Indians": "MI", "Chennai Super Kings": "CSK",
    "Royal Challengers Bangalore": "RCB", "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC", "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH", "Punjab Kings": "PBKS",
    "Lucknow Super Giants": "LSG", "Gujarat Titans": "GT"
}

# Team strength ratings for IPL (0-100)
IPL_TEAM_STRENGTH = {
    "Mumbai Indians": 88, "Chennai Super Kings": 90,
    "Royal Challengers Bangalore": 82, "Kolkata Knight Riders": 85,
    "Delhi Capitals": 80, "Rajasthan Royals": 83,
    "Sunrisers Hyderabad": 78, "Punjab Kings": 75,
    "Lucknow Super Giants": 81, "Gujarat Titans": 86
}

# IPL Venues
IPL_VENUES = {
    "Wankhede Stadium, Mumbai": {"city": "Mumbai", "home_team": "Mumbai Indians", "pitch": "batting"},
    "M.A. Chidambaram Stadium, Chennai": {"city": "Chennai", "home_team": "Chennai Super Kings", "pitch": "spin"},
    "M. Chinnaswamy Stadium, Bangalore": {"city": "Bangalore", "home_team": "Royal Challengers Bangalore", "pitch": "batting"},
    "Eden Gardens, Kolkata": {"city": "Kolkata", "home_team": "Kolkata Knight Riders", "pitch": "balanced"},
    "Arun Jaitley Stadium, Delhi": {"city": "Delhi", "home_team": "Delhi Capitals", "pitch": "balanced"},
    "Sawai Mansingh Stadium, Jaipur": {"city": "Jaipur", "home_team": "Rajasthan Royals", "pitch": "batting"},
    "Rajiv Gandhi Stadium, Hyderabad": {"city": "Hyderabad", "home_team": "Sunrisers Hyderabad", "pitch": "balanced"},
    "IS Bindra Stadium, Mohali": {"city": "Mohali", "home_team": "Punjab Kings", "pitch": "pace"},
    "BRSABV Ekana Stadium, Lucknow": {"city": "Lucknow", "home_team": "Lucknow Super Giants", "pitch": "balanced"},
    "Narendra Modi Stadium, Ahmedabad": {"city": "Ahmedabad", "home_team": "Gujarat Titans", "pitch": "balanced"},
}

# IPL Players per team
IPL_PLAYERS = {
    "Mumbai Indians": {
        "batsmen": [
            {"name": "Rohit Sharma", "avg": 31.2, "sr": 130.5, "role": "opener"},
            {"name": "Suryakumar Yadav", "avg": 33.8, "sr": 145.2, "role": "top"},
            {"name": "Ishan Kishan", "avg": 28.6, "sr": 136.8, "role": "opener"},
            {"name": "Tilak Varma", "avg": 32.1, "sr": 138.4, "role": "middle"},
            {"name": "Tim David", "avg": 25.3, "sr": 158.7, "role": "finisher"},
        ],
        "bowlers": [
            {"name": "Jasprit Bumrah", "avg": 23.5, "sr": 18.2, "econ": 7.4, "type": "fast"},
            {"name": "Piyush Chawla", "avg": 28.1, "sr": 20.5, "econ": 8.2, "type": "spin"},
        ]
    },
    "Chennai Super Kings": {
        "batsmen": [
            {"name": "Ruturaj Gaikwad", "avg": 35.4, "sr": 132.1, "role": "opener"},
            {"name": "Devon Conway", "avg": 33.2, "sr": 128.6, "role": "opener"},
            {"name": "Shivam Dube", "avg": 29.8, "sr": 142.3, "role": "middle"},
            {"name": "Ravindra Jadeja", "avg": 26.5, "sr": 131.8, "role": "middle"},
            {"name": "MS Dhoni", "avg": 24.1, "sr": 136.5, "role": "finisher"},
        ],
        "bowlers": [
            {"name": "Deepak Chahar", "avg": 25.8, "sr": 19.1, "econ": 7.8, "type": "fast"},
            {"name": "Ravindra Jadeja", "avg": 30.2, "sr": 22.5, "econ": 7.5, "type": "spin"},
        ]
    },
    "Royal Challengers Bangalore": {
        "batsmen": [
            {"name": "Virat Kohli", "avg": 38.2, "sr": 131.6, "role": "opener"},
            {"name": "Faf du Plessis", "avg": 32.5, "sr": 134.2, "role": "opener"},
            {"name": "Glenn Maxwell", "avg": 27.8, "sr": 154.3, "role": "middle"},
            {"name": "Rajat Patidar", "avg": 30.1, "sr": 141.5, "role": "top"},
            {"name": "Dinesh Karthik", "avg": 22.5, "sr": 148.2, "role": "finisher"},
        ],
        "bowlers": [
            {"name": "Mohammed Siraj", "avg": 26.3, "sr": 19.8, "econ": 8.1, "type": "fast"},
            {"name": "Wanindu Hasaranga", "avg": 22.1, "sr": 16.8, "econ": 7.4, "type": "spin"},
        ]
    },
    "Kolkata Knight Riders": {
        "batsmen": [
            {"name": "Venkatesh Iyer", "avg": 28.4, "sr": 128.5, "role": "opener"},
            {"name": "Sunil Narine", "avg": 18.5, "sr": 162.3, "role": "opener"},
            {"name": "Shreyas Iyer", "avg": 33.6, "sr": 127.8, "role": "top"},
            {"name": "Andre Russell", "avg": 25.2, "sr": 172.5, "role": "finisher"},
            {"name": "Rinku Singh", "avg": 31.8, "sr": 149.6, "role": "middle"},
        ],
        "bowlers": [
            {"name": "Sunil Narine", "avg": 24.5, "sr": 18.6, "econ": 6.8, "type": "spin"},
            {"name": "Varun Chakravarthy", "avg": 26.8, "sr": 20.1, "econ": 7.2, "type": "spin"},
        ]
    },
    "Delhi Capitals": {
        "batsmen": [
            {"name": "David Warner", "avg": 35.8, "sr": 142.1, "role": "opener"},
            {"name": "Jake Fraser-McGurk", "avg": 22.5, "sr": 165.8, "role": "opener"},
            {"name": "Rishabh Pant", "avg": 30.5, "sr": 148.3, "role": "middle"},
            {"name": "Tristan Stubbs", "avg": 25.8, "sr": 152.1, "role": "middle"},
            {"name": "Axar Patel", "avg": 22.1, "sr": 135.6, "role": "middle"},
        ],
        "bowlers": [
            {"name": "Anrich Nortje", "avg": 24.2, "sr": 17.5, "econ": 8.0, "type": "fast"},
            {"name": "Kuldeep Yadav", "avg": 23.8, "sr": 17.2, "econ": 7.8, "type": "spin"},
        ]
    },
    "Rajasthan Royals": {
        "batsmen": [
            {"name": "Yashasvi Jaiswal", "avg": 34.5, "sr": 151.2, "role": "opener"},
            {"name": "Jos Buttler", "avg": 36.2, "sr": 147.5, "role": "opener"},
            {"name": "Sanju Samson", "avg": 29.8, "sr": 138.4, "role": "top"},
            {"name": "Riyan Parag", "avg": 24.3, "sr": 141.2, "role": "middle"},
            {"name": "Shimron Hetmyer", "avg": 25.1, "sr": 155.8, "role": "finisher"},
        ],
        "bowlers": [
            {"name": "Trent Boult", "avg": 25.3, "sr": 18.9, "econ": 7.9, "type": "fast"},
            {"name": "Yuzvendra Chahal", "avg": 24.1, "sr": 17.5, "econ": 7.6, "type": "spin"},
        ]
    },
    "Sunrisers Hyderabad": {
        "batsmen": [
            {"name": "Travis Head", "avg": 32.5, "sr": 155.3, "role": "opener"},
            {"name": "Abhishek Sharma", "avg": 26.8, "sr": 148.2, "role": "opener"},
            {"name": "Heinrich Klaasen", "avg": 35.2, "sr": 171.5, "role": "middle"},
            {"name": "Aiden Markram", "avg": 28.3, "sr": 132.1, "role": "top"},
            {"name": "Nitish Kumar Reddy", "avg": 23.5, "sr": 138.4, "role": "middle"},
        ],
        "bowlers": [
            {"name": "Bhuvneshwar Kumar", "avg": 27.5, "sr": 20.8, "econ": 7.5, "type": "fast"},
            {"name": "T Natarajan", "avg": 28.8, "sr": 21.5, "econ": 8.2, "type": "fast"},
        ]
    },
    "Punjab Kings": {
        "batsmen": [
            {"name": "Shikhar Dhawan", "avg": 34.8, "sr": 126.5, "role": "opener"},
            {"name": "Jonny Bairstow", "avg": 28.5, "sr": 141.8, "role": "opener"},
            {"name": "Liam Livingstone", "avg": 24.2, "sr": 155.3, "role": "middle"},
            {"name": "Sam Curran", "avg": 22.8, "sr": 138.5, "role": "middle"},
            {"name": "Jitesh Sharma", "avg": 21.5, "sr": 145.2, "role": "finisher"},
        ],
        "bowlers": [
            {"name": "Arshdeep Singh", "avg": 26.1, "sr": 19.5, "econ": 8.3, "type": "fast"},
            {"name": "Kagiso Rabada", "avg": 23.8, "sr": 17.2, "econ": 8.0, "type": "fast"},
        ]
    },
    "Lucknow Super Giants": {
        "batsmen": [
            {"name": "KL Rahul", "avg": 37.5, "sr": 133.2, "role": "opener"},
            {"name": "Quinton de Kock", "avg": 32.1, "sr": 141.5, "role": "opener"},
            {"name": "Nicholas Pooran", "avg": 26.5, "sr": 156.8, "role": "middle"},
            {"name": "Marcus Stoinis", "avg": 24.8, "sr": 148.3, "role": "middle"},
            {"name": "Ayush Badoni", "avg": 23.2, "sr": 138.5, "role": "middle"},
        ],
        "bowlers": [
            {"name": "Mark Wood", "avg": 25.5, "sr": 18.2, "econ": 8.5, "type": "fast"},
            {"name": "Ravi Bishnoi", "avg": 27.2, "sr": 20.5, "econ": 7.5, "type": "spin"},
        ]
    },
    "Gujarat Titans": {
        "batsmen": [
            {"name": "Shubman Gill", "avg": 36.8, "sr": 139.2, "role": "opener"},
            {"name": "Wriddhiman Saha", "avg": 27.5, "sr": 131.8, "role": "opener"},
            {"name": "Hardik Pandya", "avg": 28.3, "sr": 147.5, "role": "middle"},
            {"name": "David Miller", "avg": 29.1, "sr": 142.8, "role": "finisher"},
            {"name": "Rahul Tewatia", "avg": 20.5, "sr": 135.2, "role": "finisher"},
        ],
        "bowlers": [
            {"name": "Mohammed Shami", "avg": 24.8, "sr": 18.5, "econ": 7.8, "type": "fast"},
            {"name": "Rashid Khan", "avg": 20.5, "sr": 15.8, "econ": 6.5, "type": "spin"},
        ]
    },
}

# IPL Seasons data
IPL_SEASONS = list(range(2018, 2027))


def generate_ipl_data(n_matches=2000):
    """Generate realistic IPL match data."""
    np.random.seed(789)
    records = []

    for i in range(n_matches):
        # Pick two different teams
        team1, team2 = np.random.choice(IPL_TEAMS, size=2, replace=False)

        # Pick venue (weighted towards home venues)
        home_venues = [v for v, info in IPL_VENUES.items()
                       if info["home_team"] in [team1, team2]]
        other_venues = [v for v in IPL_VENUES.keys() if v not in home_venues]

        if home_venues and np.random.random() < 0.6:
            venue = np.random.choice(home_venues)
        else:
            venue = np.random.choice(list(IPL_VENUES.keys()))

        venue_info = IPL_VENUES[venue]
        pitch_type = venue_info["pitch"]

        # Home advantage
        is_team1_home = 1 if venue_info["home_team"] == team1 else 0
        is_team2_home = 1 if venue_info["home_team"] == team2 else 0

        # Season
        season = np.random.choice(IPL_SEASONS)

        # Toss
        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(["bat", "field"], p=[0.35, 0.65])

        # Match phase (league / playoff / final)
        match_phase = np.random.choice(["league", "playoff", "final"], p=[0.85, 0.10, 0.05])

        # Calculate win probability
        strength_diff = IPL_TEAM_STRENGTH[team1] - IPL_TEAM_STRENGTH[team2]
        home_advantage = (is_team1_home - is_team2_home) * 6
        toss_advantage = 2 if toss_winner == team1 else -2

        # Pitch factor
        pitch_factor = 0
        if pitch_type == "batting":
            # Batting pitches favor batting-heavy teams
            if IPL_TEAM_STRENGTH[team1] > 83:
                pitch_factor += 3
            if IPL_TEAM_STRENGTH[team2] > 83:
                pitch_factor -= 3

        # Playoff pressure (stronger teams handle pressure better)
        if match_phase in ["playoff", "final"]:
            pitch_factor += (IPL_TEAM_STRENGTH[team1] - IPL_TEAM_STRENGTH[team2]) * 0.2

        logit = (strength_diff + home_advantage + toss_advantage + pitch_factor) / 25
        prob_team1_wins = 1 / (1 + np.exp(-logit))
        prob_team1_wins += np.random.normal(0, 0.08)
        prob_team1_wins = np.clip(prob_team1_wins, 0.15, 0.85)

        winner = team1 if np.random.random() < prob_team1_wins else team2

        # Generate T20 scores
        base_score = np.random.normal(170, 22)
        if pitch_type == "batting":
            base_score += 12
        elif pitch_type == "pace":
            base_score -= 8
        elif pitch_type == "spin":
            base_score -= 5

        team1_score = int(max(90, base_score + np.random.normal(0, 18)))
        if winner == team1:
            team2_score = int(max(80, team1_score - np.random.randint(3, 45)))
        else:
            team2_score = int(max(team1_score + 1, team1_score + np.random.randint(1, 30)))

        # Player of the match
        winner_players = IPL_PLAYERS.get(winner, {})
        potm_pool = winner_players.get("batsmen", []) + winner_players.get("bowlers", [])
        player_of_match = np.random.choice([p["name"] for p in potm_pool]) if potm_pool else "Unknown"

        # Win margin
        if winner == team1:
            if np.random.random() < 0.5:
                win_margin = f"{np.random.randint(1, 8)} wickets"
            else:
                win_margin = f"{np.random.randint(2, 50)} runs"
        else:
            if np.random.random() < 0.5:
                win_margin = f"{np.random.randint(1, 8)} wickets"
            else:
                win_margin = f"{np.random.randint(2, 50)} runs"

        records.append({
            "match_id": i + 1,
            "season": season,
            "match_phase": match_phase,
            "team1": team1,
            "team2": team2,
            "team1_short": IPL_TEAM_SHORT[team1],
            "team2_short": IPL_TEAM_SHORT[team2],
            "venue": venue,
            "city": venue_info["city"],
            "pitch_type": pitch_type,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "team1_home": is_team1_home,
            "team2_home": is_team2_home,
            "team1_strength": IPL_TEAM_STRENGTH[team1],
            "team2_strength": IPL_TEAM_STRENGTH[team2],
            "team1_score": team1_score,
            "team2_score": team2_score,
            "winner": winner,
            "winner_short": IPL_TEAM_SHORT[winner],
            "win_margin": win_margin,
            "player_of_match": player_of_match,
        })

    return pd.DataFrame(records)


def main():
    """Generate and save IPL dataset."""
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("🏏 Generating IPL Match Data...")
    ipl_df = generate_ipl_data(2000)
    ipl_df.to_csv(os.path.join(data_dir, "ipl.csv"), index=False)
    print(f"   ✅ Generated {len(ipl_df)} IPL match records")
    print(f"   Columns: {list(ipl_df.columns)}")
    print(f"   Seasons: {sorted(ipl_df['season'].unique())}")
    print(f"\n📊 Winner distribution:")
    for team, count in ipl_df["winner"].value_counts().items():
        print(f"   {team}: {count} wins")


if __name__ == "__main__":
    main()
