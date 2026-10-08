"""
Streamlit Web Application: Football Player Performance Prediction & Rating System
AIML Mini Project (Experiment 12)

A clean, beginner-friendly web interface allowing users to input player match statistics
and receive an instantaneous Machine Learning performance rating and category.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Add src to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from predict import predict_player_rating, load_prediction_bundle, get_performance_category

# Page configuration
st.set_page_config(
    page_title="Football Player Rating System | AIML Mini Project",
    page_icon="⚽",
    layout="wide"
)

# Load trained model bundle with caching
@st.cache_resource
def get_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "player_rating_model.joblib"))
    return load_prediction_bundle(model_path)

try:
    bundle = get_model()
except Exception as e:
    st.error(f"Error loading trained model: {e}. Please ensure 'models/player_rating_model.joblib' exists.")
    st.stop()

# Header
st.title("⚽ Football Player Performance Prediction & Rating System")
st.markdown(
    """
    **AIML Mini Project (Experiment 12)** — *Supervised Machine Learning (Regression)*  
    Predict a player's season performance rating (scale 1.0 – 10.0) based on match statistics.
    """
)
st.write("---")

# Quick Demo Profiles in Sidebar
st.sidebar.header("📋 Quick Demo Profiles")
st.sidebar.info("Select a preset profile to quickly fill the form for a live viva demonstration:")

preset = st.sidebar.selectbox(
    "Choose Preset Profile:",
    [
        "Custom Input (Manual)",
        "Profile 1: Star Forward (Elite Attacker)",
        "Profile 2: Playmaker Midfielder (Creative Engine)",
        "Profile 3: Rock-Solid Defender (High Defensive Volume)",
        "Profile 4: Struggling / Squad Player"
    ]
)

# Preset stats dictionary
presets_dict = {
    "Custom Input (Manual)": {
        "Age": 24, "Position": "Forward", "Matches_Played": 28, "Minutes_Played": 2300,
        "Goals": 14, "Assists": 6, "Pass_Accuracy": 78.5, "Shots": 65, "Key_Passes": 32,
        "Tackles": 18, "Interceptions": 10, "Distance_Covered": 10.2, "Sprint_Speed": 33.2
    },
    "Profile 1: Star Forward (Elite Attacker)": {
        "Age": 25, "Position": "Forward", "Matches_Played": 34, "Minutes_Played": 2950,
        "Goals": 26, "Assists": 8, "Pass_Accuracy": 81.0, "Shots": 105, "Key_Passes": 48,
        "Tackles": 14, "Interceptions": 8, "Distance_Covered": 10.1, "Sprint_Speed": 34.6
    },
    "Profile 2: Playmaker Midfielder (Creative Engine)": {
        "Age": 27, "Position": "Midfielder", "Matches_Played": 35, "Minutes_Played": 3050,
        "Goals": 8, "Assists": 16, "Pass_Accuracy": 89.0, "Shots": 38, "Key_Passes": 78,
        "Tackles": 46, "Interceptions": 38, "Distance_Covered": 11.9, "Sprint_Speed": 31.0
    },
    "Profile 3: Rock-Solid Defender (High Defensive Volume)": {
        "Age": 28, "Position": "Defender", "Matches_Played": 33, "Minutes_Played": 2900,
        "Goals": 2, "Assists": 1, "Pass_Accuracy": 85.5, "Shots": 8, "Key_Passes": 12,
        "Tackles": 88, "Interceptions": 74, "Distance_Covered": 9.9, "Sprint_Speed": 32.1
    },
    "Profile 4: Struggling / Squad Player": {
        "Age": 22, "Position": "Winger", "Matches_Played": 16, "Minutes_Played": 980,
        "Goals": 1, "Assists": 1, "Pass_Accuracy": 68.0, "Shots": 18, "Key_Passes": 8,
        "Tackles": 6, "Interceptions": 4, "Distance_Covered": 9.2, "Sprint_Speed": 29.5
    }
}

current_vals = presets_dict[preset]

# Main Interface Tabs
tab_predict, tab_analytics, tab_project_info = st.tabs(["🎯 Player Rating Predictor", "📊 Visualizations & Model Evaluation", "📖 Project Documentation"])

with tab_predict:
    st.subheader("Enter Player Match Statistics")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 General Profile")
            pos_options = ["Forward", "Midfielder", "Defender", "Winger"]
            pos_index = pos_options.index(current_vals["Position"]) if current_vals["Position"] in pos_options else 0
            position = st.selectbox("Primary Position:", pos_options, index=pos_index)
            age = st.slider("Age (Years):", 17, 40, int(current_vals["Age"]))
            matches = st.number_input("Matches Played:", min_value=1, max_value=50, value=int(current_vals["Matches_Played"]))
            minutes = st.number_input("Minutes Played:", min_value=10, max_value=4500, value=int(current_vals["Minutes_Played"]))
            
        with col2:
            st.markdown("#### ⚡ Attacking & Playmaking")
            goals = st.number_input("Goals Scored:", min_value=0, max_value=60, value=int(current_vals["Goals"]))
            assists = st.number_input("Assists Provided:", min_value=0, max_value=40, value=int(current_vals["Assists"]))
            shots = st.number_input("Total Shots:", min_value=0, max_value=200, value=int(current_vals["Shots"]))
            key_passes = st.number_input("Key Passes (Chances Created):", min_value=0, max_value=150, value=int(current_vals["Key_Passes"]))
            pass_accuracy = st.slider("Pass Accuracy (%):", 50.0, 100.0, float(current_vals["Pass_Accuracy"]), step=0.5)

        with col3:
            st.markdown("#### 🛡️ Defending & Physical")
            tackles = st.number_input("Tackles Won:", min_value=0, max_value=150, value=int(current_vals["Tackles"]))
            interceptions = st.number_input("Interceptions Made:", min_value=0, max_value=120, value=int(current_vals["Interceptions"]))
            distance = st.slider("Avg Distance Covered (km/match):", 7.0, 14.0, float(current_vals["Distance_Covered"]), step=0.1)
            speed = st.slider("Top Sprint Speed (km/h):", 24.0, 38.0, float(current_vals["Sprint_Speed"]), step=0.1)
            
        submitted = st.form_submit_button("⚽ Predict Performance Rating", use_container_width=True)

    # Prediction Outcome Section
    if submitted or preset != "Custom Input (Manual)":
        player_dict = {
            "Age": age,
            "Position": position,
            "Matches_Played": matches,
            "Minutes_Played": minutes,
            "Goals": goals,
            "Assists": assists,
            "Pass_Accuracy": pass_accuracy,
            "Shots": shots,
            "Key_Passes": key_passes,
            "Tackles": tackles,
            "Interceptions": interceptions,
            "Distance_Covered": distance,
            "Sprint_Speed": speed
        }
        
        result = predict_player_rating(player_dict, bundle=bundle)
        rating = result["predicted_rating"]
        cat = result["category"]
        cat_desc = result["category_description"]
        summary = result["summary"]
        
        # Color badge logic
        color_map = {
            "Excellent": "#28a745", # Green
            "Good": "#17a2b8",      # Cyan
            "Average": "#ffc107",   # Yellow/Orange
            "Below Average": "#dc3545" # Red
        }
        theme_color = color_map.get(cat, "#007bff")
        
        st.write("---")
        st.subheader("🎯 Prediction Result")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border: 2px solid {theme_color}; border-radius: 10px; padding: 20px; text-align: center;">
                    <h4 style="margin: 0; color: #6c757d;">Predicted Performance Rating</h4>
                    <h1 style="font-size: 52px; margin: 10px 0; color: {theme_color};">{rating:.1f} <span style="font-size: 24px; color: #6c757d;">/ 10</span></h1>
                    <div style="background-color: {theme_color}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; display: inline-block;">
                        {cat}
                    </div>
                    <p style="margin-top: 10px; color: #495057; font-size: 14px;"><em>{cat_desc}</em></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with res_col2:
            st.markdown("#### 📝 Scouting Report & Explanation")
            st.info(summary)
            
            # Progress bar comparing to dataset average (7.26)
            st.markdown(f"**Rating vs Benchmark:**")
            st.progress(float(rating / 10.0))
            
            diff = rating - 7.26
            diff_text = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
            st.caption(f"ℹ️ Current rating is **{diff_text}** points relative to the dataset average (7.26 / 10.0). Model: **{result['model_used']}**.")

with tab_analytics:
    st.subheader("📊 Model Performance & Viva Visualizations")
    st.write("These graphs were generated from the actual trained model and test evaluation dataset.")
    
    # Metrics table
    lr_m = bundle["lr_metrics"]
    rf_m = bundle["rf_metrics"]
    
    metrics_df = pd.DataFrame({
        "Metric": ["MAE (Mean Absolute Error)", "MSE (Mean Squared Error)", "RMSE (Root Mean Squared Error)", "R² Score (Accuracy)"],
        "Linear Regression (Baseline)": [f"{lr_m['MAE']:.4f}", f"{lr_m['MSE']:.4f}", f"{lr_m['RMSE']:.4f}", f"{lr_m['R2']:.4f} ({lr_m['R2']*100:.2f}%)"],
        "Random Forest Regressor": [f"{rf_m['MAE']:.4f}", f"{rf_m['MSE']:.4f}", f"{rf_m['RMSE']:.4f}", f"{rf_m['R2']:.4f} ({rf_m['R2']*100:.2f}%)"],
    })
    st.table(metrics_df)
    
    # Image display
    img_col1, img_col2 = st.columns(2)
    vis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "visualizations"))
    
    with img_col1:
        p1 = os.path.join(vis_dir, "1_rating_distribution.png")
        if os.path.exists(p1):
            st.image(p1, caption="1. Distribution of Football Player Ratings (N=1000)")
        p3 = os.path.join(vis_dir, "3_assists_vs_rating.png")
        if os.path.exists(p3):
            st.image(p3, caption="3. Assists vs Performance Rating by Position")
        p5 = os.path.join(vis_dir, "5_actual_vs_predicted.png")
        if os.path.exists(p5):
            st.image(p5, caption="5. Actual vs Predicted Ratings on Test Set")
            
    with img_col2:
        p2 = os.path.join(vis_dir, "2_goals_vs_rating.png")
        if os.path.exists(p2):
            st.image(p2, caption="2. Goals vs Performance Rating by Position")
        p4 = os.path.join(vis_dir, "4_feature_importance.png")
        if os.path.exists(p4):
            st.image(p4, caption="4. Random Forest Feature Importance")
        p6 = os.path.join(vis_dir, "6_model_comparison.png")
        if os.path.exists(p6):
            st.image(p6, caption="6. Model Comparison (MAE, RMSE, R²)")

with tab_project_info:
    st.subheader("📖 Project Overview (Experiment 12)")
    st.markdown(
        """
        ### Problem Statement
        In modern association football, evaluating a player's seasonal contribution requires objective, data-driven methodology rather than subjective observation alone. This project builds a machine learning regression system to predict an overall player performance rating on a 1.0 to 10.0 scale based on multidimensional match metrics.

        ### Machine Learning Methodology
        1. **Data Preprocessing:** Handled missing values, encoded categorical positions using One-Hot Encoding, and standardized numerical features with `StandardScaler`.
        2. **Algorithms Evaluated:**
           - **Linear Regression (Baseline):** Supervised regression establishing direct linear weights across features ($R^2 = 83.66\%$, $\text{MAE} = 0.29$).
           - **Random Forest Regressor:** Non-linear tree ensemble verifying feature interactions and providing feature importance rankings.
        3. **Target Variable:** `Performance_Rating` ($1.0 - 10.0$), mapped to 4 performance categories:
           - **Excellent:** $\ge 8.5$
           - **Good:** $7.0 - 8.4$
           - **Average:** $5.5 - 6.9$
           - **Below Average:** $< 5.5$
        """
    )
    
    st.info("🎓 Developed as an AIML Mini Project (Experiment 12) for 3rd-year Engineering.")
