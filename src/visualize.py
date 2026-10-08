"""
Visualization Script: Football Player Performance Rating System
Academic Mini Project - Experiment 12

Generates 6 clean, high-resolution plots for reports and viva presentations:
1. Distribution of Player Ratings
2. Goals vs Performance Rating (by Position)
3. Assists vs Performance Rating (by Position)
4. Random Forest Feature Importance
5. Actual vs Predicted Ratings (Test Set)
6. Model Comparison (MAE, RMSE, R2 Score)
"""

import os
import sys
import warnings
# Suppress benign warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure clean plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.autolayout": True,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

def generate_visualizations(
    dataset_path="dataset/football_players.csv", 
    model_bundle_path="models/player_rating_model.joblib",
    output_dir="visualizations"
):
    print("=" * 65)
    print(" STEP 5: GENERATING ACADEMIC VISUALIZATIONS")
    print("=" * 65)
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)
    bundle = joblib.load(model_bundle_path)
    
    # 1. Distribution of Player Ratings
    print("\n1. Generating: Rating Distribution Plot...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(df["Performance_Rating"], kde=True, bins=25, color="#1f77b4", ax=ax, edgecolor="black")
    mean_val = df["Performance_Rating"].mean()
    median_val = df["Performance_Rating"].median()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle=":", linewidth=2, label=f"Median: {median_val:.2f}")
    ax.set_title("Distribution of Football Player Ratings (N = 1000)", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Performance Rating (Scale: 1.0 - 10.0)", fontsize=12)
    ax.set_ylabel("Player Count", fontsize=12)
    ax.legend(frameon=True)
    fig.savefig(os.path.join(output_dir, "1_rating_distribution.png"))
    plt.close(fig)
    print("   ✓ Saved to visualizations/1_rating_distribution.png")

    # 2. Goals vs Performance Rating
    print("2. Generating: Goals vs Rating Plot...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    palette = {"Forward": "#d62728", "Midfielder": "#1f77b4", "Defender": "#2ca02c", "Winger": "#ff7f0e"}
    sns.scatterplot(data=df, x="Goals", y="Performance_Rating", hue="Position", palette=palette, alpha=0.8, s=50, ax=ax)
    sns.regplot(data=df, x="Goals", y="Performance_Rating", scatter=False, color="black", ax=ax, line_kws={"linestyle": "--", "linewidth": 1.5, "label": "Overall Trend"})
    ax.set_title("Season Goals vs Player Performance Rating", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Goals Scored in Season", fontsize=12)
    ax.set_ylabel("Performance Rating (1.0 - 10.0)", fontsize=12)
    ax.legend(title="Position", frameon=True)
    fig.savefig(os.path.join(output_dir, "2_goals_vs_rating.png"))
    plt.close(fig)
    print("   ✓ Saved to visualizations/2_goals_vs_rating.png")

    # 3. Assists vs Performance Rating
    print("3. Generating: Assists vs Rating Plot...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.scatterplot(data=df, x="Assists", y="Performance_Rating", hue="Position", palette=palette, alpha=0.8, s=50, ax=ax)
    sns.regplot(data=df, x="Assists", y="Performance_Rating", scatter=False, color="black", ax=ax, line_kws={"linestyle": "--", "linewidth": 1.5, "label": "Overall Trend"})
    ax.set_title("Season Assists vs Player Performance Rating", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Assists Provided in Season", fontsize=12)
    ax.set_ylabel("Performance Rating (1.0 - 10.0)", fontsize=12)
    ax.legend(title="Position", frameon=True)
    fig.savefig(os.path.join(output_dir, "3_assists_vs_rating.png"))
    plt.close(fig)
    print("   ✓ Saved to visualizations/3_assists_vs_rating.png")

    # 4. Feature Importance for Random Forest
    print("4. Generating: Random Forest Feature Importance Plot...")
    rf_model = bundle["rf_model"]
    preprocessor = bundle["preprocessor"]
    num_cols = bundle["numerical_features"]
    cat_cols = list(preprocessor.named_transformers_['cat'].get_feature_names_out(bundle["categorical_features"]))
    all_features = num_cols + cat_cols
    importances = rf_model.feature_importances_
    
    feat_df = pd.DataFrame({"Feature": all_features, "Importance": importances}).sort_values(by="Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    colors = ["#4c72b0" if "Position" not in f else "#55a868" for f in feat_df["Feature"]]
    ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="black")
    ax.set_title("Random Forest: Relative Feature Importance", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Gini Feature Importance Score", fontsize=12)
    fig.savefig(os.path.join(output_dir, "4_feature_importance.png"))
    plt.close(fig)
    print("   ✓ Saved to visualizations/4_feature_importance.png")

    # 5. Actual vs Predicted Ratings (Test Set)
    print("5. Generating: Actual vs Predicted Ratings Plot...")
    y_test = bundle["y_test"]
    lr_preds = bundle["lr_preds"]
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.scatter(y_test, lr_preds, color="#2b5c8f", alpha=0.75, edgecolors="k", s=55, label="Test Set Players (N=200)")
    # 45-degree reference line
    min_val = min(y_test.min(), lr_preds.min()) - 0.2
    max_val = max(y_test.max(), lr_preds.max()) + 0.2
    ax.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=2.5, label="Perfect Prediction (y = x)")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_title("Linear Regression: Actual vs Predicted Ratings (Test Set)", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Actual Performance Rating", fontsize=12)
    ax.set_ylabel("Predicted Performance Rating", fontsize=12)
    ax.legend(frameon=True)
    fig.savefig(os.path.join(output_dir, "5_actual_vs_predicted.png"))
    plt.close(fig)
    print("   ✓ Saved to visualizations/5_actual_vs_predicted.png")

    # 6. Model Evaluation Metrics Comparison
    print("6. Generating: Model Comparison Bar Chart...")
    lr_m = bundle["lr_metrics"]
    rf_m = bundle["rf_metrics"]
    
    metrics_names = ["MAE (Rating pts)", "RMSE (Rating pts)", "R² Score"]
    lr_vals = [lr_m["MAE"], lr_m["RMSE"], lr_m["R2"]]
    rf_vals = [rf_m["MAE"], rf_m["RMSE"], rf_m["R2"]]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - width/2, lr_vals, width, label="Linear Regression", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x + width/2, rf_vals, width, label="Random Forest", color="#ff7f0e", edgecolor="black")
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Comparison of Model Performance Metrics", fontsize=14, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True)
    fig.savefig(os.path.join(output_dir, "6_model_comparison.png"))
    plt.close(fig)
    print("   ✓ Saved to visualizations/6_model_comparison.png")
    
    print("\n" + "=" * 65)
    print("All 6 visualization plots saved successfully to 'visualizations/' directory.")
    print("=" * 65)

if __name__ == "__main__":
    generate_visualizations()
