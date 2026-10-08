"""
Prediction System: Football Player Performance Rating System
Academic Mini Project - Experiment 12

This module provides a standalone prediction function and CLI demonstration.
It loads the saved joblib model bundle, processes player input features,
and returns the exact predicted rating, category, and an explainable summary.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

import joblib
import pandas as pd
import numpy as np

# Position normalizer to handle common football abbreviations
POSITION_MAP = {
    "FORWARD": "Forward",
    "FW": "Forward",
    "ST": "Forward",
    "CF": "Forward",
    "MIDFIELDER": "Midfielder",
    "MID": "Midfielder",
    "MF": "Midfielder",
    "CM": "Midfielder",
    "CAM": "Midfielder",
    "CDM": "Midfielder",
    "DEFENDER": "Defender",
    "DEF": "Defender",
    "CB": "Defender",
    "LB": "Defender",
    "RB": "Defender",
    "WINGER": "Winger",
    "RW": "Winger",
    "LW": "Winger",
    "RM": "Winger",
    "LM": "Winger"
}

def load_prediction_bundle(bundle_path="models/player_rating_model.joblib"):
    """Loads the trained model and preprocessor bundle."""
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Trained model not found at '{bundle_path}'. Please run 'python3 src/train_model.py' first."
        )
    return joblib.load(bundle_path)

def get_performance_category(rating):
    """Maps a continuous rating (1.0 to 10.0) to an intuitive qualitative tier."""
    if rating >= 8.5:
        return "Excellent", "Star / League-Leading Performer"
    elif rating >= 7.0:
        return "Good", "Quality First-Team Starter"
    elif rating >= 5.5:
        return "Average", "Solid Squad / Rotation Player"
    else:
        return "Below Average", "Underperforming / Needs Improvement"

def generate_scouting_summary(player_data, rating, category):
    """Generates a brief football analytics summary explaining the rating."""
    pos = player_data["Position"]
    matches = max(1, player_data["Matches_Played"])
    goals = player_data["Goals"]
    assists = player_data["Assists"]
    tackles = player_data["Tackles"]
    pass_acc = player_data["Pass_Accuracy"]
    
    highlights = []
    if pos in ["Forward", "Winger"]:
        g_pm = goals / matches
        if g_pm >= 0.5:
            highlights.append(f"outstanding goal return ({goals} goals in {matches} matches)")
        elif g_pm >= 0.25:
            highlights.append(f"steady goal threat ({goals} goals)")
        if assists >= 5:
            highlights.append(f"strong creative support ({assists} assists)")
    elif pos == "Midfielder":
        if assists >= 6:
            highlights.append(f"superb playmaking vision ({assists} assists)")
        if pass_acc >= 85:
            highlights.append(f"elite ball retention ({pass_acc:.1f}% pass accuracy)")
        if tackles >= 25:
            highlights.append(f"reliable defensive work rate ({tackles} tackles)")
    elif pos == "Defender":
        if tackles >= 40:
            highlights.append(f"commanding tackle volume ({tackles} tackles won)")
        if player_data.get("Interceptions", 0) >= 30:
            highlights.append("sharp positional anticipation")
        if pass_acc >= 82:
            highlights.append("composed build-up distribution from the back")
            
    if not highlights:
        highlights.append(f"balanced contributions across {matches} appearances")
        
    summary = (
        f"Based on the entered statistics, the model predicts an overall rating of {rating:.1f}/10 ({category}). "
        f"Key drivers include {', '.join(highlights)}."
    )
    return summary

def predict_player_rating(player_dict, bundle=None, bundle_path="models/player_rating_model.joblib"):
    """
    Predicts the performance rating for a single football player.
    
    Args:
        player_dict (dict): Dictionary with all required player statistics.
        bundle (dict, optional): Pre-loaded model bundle.
        bundle_path (str): Path to load bundle if not passed.
        
    Returns:
        dict: {
            'predicted_rating': float,
            'category': str,
            'category_description': str,
            'summary': str,
            'model_used': str
        }
    """
    if bundle is None:
        bundle = load_prediction_bundle(bundle_path)
        
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    num_features = bundle["numerical_features"]
    cat_features = bundle["categorical_features"]
    
    # Normalize position string
    raw_pos = str(player_dict.get("Position", "Midfielder")).strip()
    norm_pos = POSITION_MAP.get(raw_pos.upper(), raw_pos)
    if norm_pos not in ["Forward", "Midfielder", "Defender", "Winger"]:
        norm_pos = "Midfielder" # Fallback default
        
    # Build a clean 1-row DataFrame matching the exact training columns
    input_row = {
        "Age": int(player_dict.get("Age", 24)),
        "Matches_Played": int(player_dict.get("Matches_Played", 20)),
        "Minutes_Played": int(player_dict.get("Minutes_Played", 1600)),
        "Goals": int(player_dict.get("Goals", 0)),
        "Assists": int(player_dict.get("Assists", 0)),
        "Pass_Accuracy": float(player_dict.get("Pass_Accuracy", 78.0)),
        "Shots": int(player_dict.get("Shots", 10)),
        "Key_Passes": int(player_dict.get("Key_Passes", 10)),
        "Tackles": int(player_dict.get("Tackles", 15)),
        "Interceptions": int(player_dict.get("Interceptions", 10)),
        "Distance_Covered": float(player_dict.get("Distance_Covered", 10.0)),
        "Sprint_Speed": float(player_dict.get("Sprint_Speed", 31.0)),
        "Position": norm_pos
    }
    
    df_single = pd.DataFrame([input_row])[num_features + cat_features]
    
    # Transform using saved preprocessor (scales numbers & encodes position)
    X_transformed = preprocessor.transform(df_single)
    
    # Predict rating using the trained ML model
    raw_pred = float(model.predict(X_transformed)[0])
    predicted_rating = round(float(np.clip(raw_pred, 1.0, 10.0)), 1)
    
    cat, cat_desc = get_performance_category(predicted_rating)
    summary = generate_scouting_summary(input_row, predicted_rating, cat)
    
    return {
        "predicted_rating": predicted_rating,
        "category": cat,
        "category_description": cat_desc,
        "summary": summary,
        "model_used": bundle.get("model_name", "Linear Regression"),
        "normalized_position": norm_pos
    }

if __name__ == "__main__":
    print("=" * 65)
    print(" STEP 6: TESTING PREDICTION SYSTEM WITH TEST PROFILES")
    print("=" * 65)
    
    bundle = load_prediction_bundle()
    
    # Test case from prompt instructions:
    # Age=20, Position=RW/Winger, Matches=20, Minutes=1500, Goals=8, Assists=7,
    # Pass_Accuracy=84, Shots=35, Key_Passes=42, Tackles=20, Interceptions=10,
    # Distance_Covered=10.2, Sprint_Speed=28
    prompt_test_player = {
        "Age": 20,
        "Position": "RW",
        "Matches_Played": 20,
        "Minutes_Played": 1500,
        "Goals": 8,
        "Assists": 7,
        "Pass_Accuracy": 84.0,
        "Shots": 35,
        "Key_Passes": 42,
        "Tackles": 20,
        "Interceptions": 10,
        "Distance_Covered": 10.2,
        "Sprint_Speed": 28.0
    }
    
    result = predict_player_rating(prompt_test_player, bundle=bundle)
    
    print("\n[PROMPT TEST PLAYER (Young RW)]")
    print(f"Inputs: {prompt_test_player}")
    print(f"\n>> PREDICTED PERFORMANCE RATING : {result['predicted_rating']} / 10.0")
    print(f">> PERFORMANCE CATEGORY         : {result['category']} ({result['category_description']})")
    print(f">> MODEL USED                   : {result['model_used']}")
    print(f">> SCOUTING SUMMARY             : {result['summary']}")
    print("=" * 65)
