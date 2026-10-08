"""
Dataset Generation Script: Football Player Performance
Academic Mini Project - Educational Demonstration Dataset

This script generates a realistic, internally consistent dataset of 1,000 football players
across 4 positions (Forward, Midfielder, Defender, Winger) with realistic statistical
distributions and position-weighted performance ratings.
"""

import numpy as np
import pandas as pd
import os

# Set seed for reproducible academic results
np.random.seed(42)

def generate_player_dataset(num_records=1000):
    first_names = [
        "Alex", "Marco", "Lucas", "David", "Leo", "Mateo", "Julian", "Gabriel", 
        "Kevin", "Marcus", "Kylian", "Bruno", "Luka", "Thomas", "Harry", "Declan",
        "Rodri", "Jude", "Vinicius", "Bukayo", "Florian", "Jamal", "Cole", "Martin",
        "Ruben", "Virgil", "Alisson", "Ederson", "Manuel", "Joshua", "Federico", "Rafael"
    ]
    last_names = [
        "Silva", "Santos", "Fernandez", "Rodriguez", "Hernandez", "Smith", "Jones",
        "Williams", "Davies", "Mueller", "Schmidt", "Becker", "Rossi", "Mancini",
        "Costa", "Alvarez", "Martinez", "Diallo", "Traore", "Mendes", "Sane", "Gomez",
        "Kim", "Tanaka", "Patel", "Sharma", "Bellingham", "Saka", "Foden", "Palmer"
    ]
    
    positions = ["Forward", "Midfielder", "Defender", "Winger"]
    pos_probs = [0.25, 0.35, 0.25, 0.15]
    
    records = []
    
    for i in range(1, num_records + 1):
        pid = f"PLY{i:04d}"
        name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
        age = int(np.random.choice(np.arange(18, 38), p=np.array([
            0.02, 0.04, 0.06, 0.08, 0.09, 0.10, 0.10, 0.09, 0.08, 0.07,
            0.06, 0.05, 0.04, 0.04, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005
        ])))
        position = np.random.choice(positions, p=pos_probs)
        matches = np.random.randint(10, 39)
        avg_mins = np.random.uniform(50, 90)
        minutes = int(matches * avg_mins)
        
        # Player ability tier (0.45=struggling, 0.75=below avg, 1.0=average starter, 1.25=good, 1.55=star, 1.85=elite)
        ability = np.random.choice([0.45, 0.75, 1.0, 1.25, 1.55, 1.85], p=[0.06, 0.24, 0.38, 0.18, 0.09, 0.05])
        
        if position == "Forward":
            goals = np.random.poisson(lam=max(0.05, matches * 0.30 * ability))
            assists = np.random.poisson(lam=max(0.05, matches * 0.14 * ability))
            shots = int(goals * np.random.uniform(2.5, 4.0) + np.random.randint(4, 18))
            key_passes = np.random.poisson(lam=max(0.3, matches * 0.6 * ability))
            tackles = np.random.poisson(lam=max(0.2, matches * 0.35))
            interceptions = np.random.poisson(lam=max(0.1, matches * 0.25))
            pass_accuracy = np.clip(np.random.normal(74 + 5 * (ability - 1), 4.5), 58, 88)
            distance = np.clip(np.random.normal(9.6 + 0.5 * (ability - 1), 0.6), 8.0, 11.5)
            sprint_speed = np.clip(np.random.normal(32.0 + 1.2 * (ability - 1), 1.5), 28.0, 36.0)
            
            score = 5.2 + (goals / matches) * 3.2 + (assists / matches) * 2.0 + \
                    (key_passes / matches) * 0.5 + ((pass_accuracy - 70) * 0.04) + \
                    ((sprint_speed - 30) * 0.1)
                    
        elif position == "Midfielder":
            goals = np.random.poisson(lam=max(0.05, matches * 0.12 * ability))
            assists = np.random.poisson(lam=max(0.1, matches * 0.24 * ability))
            shots = int(goals * np.random.uniform(2.0, 3.5) + np.random.randint(3, 15))
            key_passes = np.random.poisson(lam=max(0.5, matches * 1.4 * ability))
            tackles = np.random.poisson(lam=max(0.5, matches * 1.3 * ability))
            interceptions = np.random.poisson(lam=max(0.3, matches * 1.0 * ability))
            pass_accuracy = np.clip(np.random.normal(82 + 5 * (ability - 1), 4.0), 65, 94)
            distance = np.clip(np.random.normal(10.8 + 0.6 * (ability - 1), 0.6), 8.8, 13.0)
            sprint_speed = np.clip(np.random.normal(30.5 + 0.8 * (ability - 1), 1.4), 26.5, 34.0)
            
            score = 4.5 + (goals / matches) * 1.8 + (assists / matches) * 2.6 + \
                    (key_passes / matches) * 0.6 + (tackles / matches) * 0.4 + \
                    ((pass_accuracy - 76) * 0.05) + ((distance - 10.0) * 0.25)
                    
        elif position == "Defender":
            goals = np.random.poisson(lam=max(0.01, matches * 0.04 * ability))
            assists = np.random.poisson(lam=max(0.01, matches * 0.06 * ability))
            shots = int(goals * np.random.uniform(1.2, 2.5) + np.random.randint(1, 8))
            key_passes = np.random.poisson(lam=max(0.1, matches * 0.35 * ability))
            tackles = np.random.poisson(lam=max(0.8, matches * 2.0 * ability))
            interceptions = np.random.poisson(lam=max(0.6, matches * 1.6 * ability))
            pass_accuracy = np.clip(np.random.normal(80 + 4 * (ability - 1), 4.5), 64, 92)
            distance = np.clip(np.random.normal(9.6 + 0.4 * (ability - 1), 0.6), 8.0, 11.5)
            sprint_speed = np.clip(np.random.normal(31.0 + 1.0 * (ability - 1), 1.5), 27.0, 35.5)
            
            score = 4.5 + (tackles / matches) * 0.55 + (interceptions / matches) * 0.65 + \
                    ((pass_accuracy - 74) * 0.045) + ((goals + assists) / matches) * 0.9 + \
                    ((sprint_speed - 29) * 0.08)
                    
        else: # Winger
            goals = np.random.poisson(lam=max(0.1, matches * 0.25 * ability))
            assists = np.random.poisson(lam=max(0.1, matches * 0.22 * ability))
            shots = int(goals * np.random.uniform(2.2, 3.8) + np.random.randint(4, 18))
            key_passes = np.random.poisson(lam=max(0.5, matches * 1.2 * ability))
            tackles = np.random.poisson(lam=max(0.2, matches * 0.55))
            interceptions = np.random.poisson(lam=max(0.1, matches * 0.35))
            pass_accuracy = np.clip(np.random.normal(77 + 4 * (ability - 1), 4.5), 62, 90)
            distance = np.clip(np.random.normal(10.2 + 0.5 * (ability - 1), 0.6), 8.5, 12.2)
            sprint_speed = np.clip(np.random.normal(33.0 + 1.2 * (ability - 1), 1.4), 29.0, 36.8)
            
            score = 4.8 + (goals / matches) * 2.4 + (assists / matches) * 2.4 + \
                    (key_passes / matches) * 0.5 + ((sprint_speed - 30) * 0.12) + \
                    ((pass_accuracy - 72) * 0.04)

        # Add realistic noise/variance so ML algorithms need to learn the underlying trend
        noise = np.random.normal(0, 0.25)
        rating = np.clip(score + noise, 4.2, 9.6)
        
        records.append({
            "Player_ID": pid,
            "Player_Name": name,
            "Age": age,
            "Position": position,
            "Matches_Played": int(matches),
            "Minutes_Played": int(minutes),
            "Goals": int(goals),
            "Assists": int(assists),
            "Pass_Accuracy": round(float(pass_accuracy), 1),
            "Shots": int(shots),
            "Key_Passes": int(key_passes),
            "Tackles": int(tackles),
            "Interceptions": int(interceptions),
            "Distance_Covered": round(float(distance), 2),
            "Sprint_Speed": round(float(sprint_speed), 1),
            "Performance_Rating": round(float(rating), 1)
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    df = generate_player_dataset(1000)
    output_path = os.path.join("dataset", "football_players.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path}")
    print(f"Shape: {df.shape}")
    print("\nSummary Statistics of Target (Performance_Rating):")
    print(df["Performance_Rating"].describe())
