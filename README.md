# Football Player Performance Prediction & Rating System using Machine Learning

**AIML Mini Project — Experiment 12**  
*Department of Computer Science & Engineering / Artificial Intelligence & Machine Learning*

---

## 1. Introduction
In modern association football (soccer), scouting and player evaluation are undergoing a significant shift from subjective visual impressions to objective, data-driven analytical modeling. Football clubs, analysts, and scouts need objective tools to assess player contributions across match appearances. 

This project develops an end-to-end Machine Learning pipeline that processes match-level and season-long performance statistics (such as goals, assists, passing accuracy, defensive actions, work rate, and sprint speed) to predict an objective **Player Performance Rating (1.0 to 10.0 scale)** and categorizes the player into actionable tiers (*Below Average*, *Average*, *Good*, *Excellent*).

---

## 2. Problem Statement
Traditional player scouting often suffers from observational bias, recency bias, and media hype. Furthermore, raw statistics can be misleading when viewed in isolation (for example, comparing a defender's low goal tally against a striker's high goal tally).

There is a need for a machine learning regression system that:
1. Normalizes and evaluates diverse football statistics.
2. Accounts for position-specific roles (Forward, Midfielder, Defender, Winger).
3. Produces a standardized, continuous performance rating with high accuracy and explainability.

---

## 3. Project Objectives
* **Data Preprocessing:** Build a robust data pipeline that checks data integrity, applies One-Hot Encoding on categorical features, and standardizes continuous numerical statistics.
* **Model Development:** Formulate the problem as a Supervised Machine Learning Regression task. Train and compare a baseline **Linear Regression** model and an ensemble **Random Forest Regressor**.
* **Model Evaluation:** Evaluate models on an unseen test set (20%) using **MAE**, **MSE**, **RMSE**, and **$R^2$ Score**.
* **Visual Analytics:** Produce publication-ready graphs illustrating rating distributions, correlations, feature importances, and error trends.
* **Interactive Demonstration:** Build an intuitive **Streamlit** web application for real-time player rating predictions and automated scouting report summaries.

---

## 4. Dataset Description
* **Dataset File:** `dataset/football_players.csv`
* **Total Records:** $1,000$ player seasons.
* **Academic Transparency:** Created as an educational football statistics dataset tailored for 3rd-year engineering demonstration, modeling realistic distributions observed in European top-flight leagues across four primary outfield positions (*Forward*, *Midfielder*, *Defender*, *Winger*).
* **Missing Values:** $0$ missing entries (clean and internally consistent).
* **Target Variable:** `Performance_Rating` (continuous float, scale $1.0 - 10.0$).

---

## 5. Input Features

The dataset comprises $16$ columns ($2$ identifiers, $13$ input features, and $1$ target variable):

| # | Feature Name | Type | Description |
|---|---|---|---|
| 1 | `Player_ID` | String | Unique ID (dropped prior to ML training) |
| 2 | `Player_Name` | String | Player name (display only) |
| 3 | `Age` | Numerical | Age in years ($18 - 37$) |
| 4 | `Position` | Categorical | Playing role (`Forward`, `Midfielder`, `Defender`, `Winger`) |
| 5 | `Matches_Played` | Numerical | Total match appearances in season ($10 - 38$) |
| 6 | `Minutes_Played` | Numerical | Total pitch minutes played |
| 7 | `Goals` | Numerical | Total goals scored |
| 8 | `Assists` | Numerical | Total goal assists registered |
| 9 | `Pass_Accuracy` | Numerical (%) | Successful pass percentage ($58\% - 94\%$) |
| 10 | `Shots` | Numerical | Total shots attempted |
| 11 | `Key_Passes` | Numerical | Passes directly leading to a teammate shot |
| 12 | `Tackles` | Numerical | Number of successful tackles won |
| 13 | `Interceptions` | Numerical | Number of opposition passes intercepted |
| 14 | `Distance_Covered` | Numerical (km) | Average distance covered per match ($8.0 - 13.0\text{ km}$) |
| 15 | `Sprint_Speed` | Numerical (km/h) | Top recorded sprint speed ($26.5 - 36.8\text{ km/h}$) |
| 16 | **`Performance_Rating`** | **Numerical (Target)** | **Continuous rating out of $10.0$** |

---

## 6. Data Preprocessing

The preprocessing pipeline (`src/preprocess.py`) follows standard scikit-learn best practices:
1. **Integrity Validation:** Checked for nulls (`df.isnull().sum()`) and duplicate rows (`df.duplicated().sum()`).
2. **Feature Separation:** Dropped non-predictive identifiers (`Player_ID`, `Player_Name`). Separated feature matrix $X$ and target vector $y$.
3. **Train-Test Split:** Split the dataset into $80\%$ Training Data ($800$ players) and $20\%$ Testing Data ($200$ players) using `random_state=42` for exact reproducibility.
4. **Encoding Categorical Features:** Transformed `Position` using `OneHotEncoder`, generating four binary indicator columns (`Position_Defender`, `Position_Forward`, `Position_Midfielder`, `Position_Winger`).
5. **Feature Scaling:** Scaled the $12$ continuous numerical features using `StandardScaler` ($z = \frac{x - \mu}{\sigma}$), fitted strictly on training data to prevent data leakage.

---

## 7. Machine Learning Algorithms

We evaluated two distinct regression algorithms:

### 1. Linear Regression (Baseline Model)
* Fits a linear hyperplane: $\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p$.
* Provides high interpretability and direct linear weighting across performance indicators.

### 2. Random Forest Regressor (Ensemble Comparison)
* Constructs an ensemble of $100$ decorrelated decision trees (`n_estimators=100`, `max_depth=12`).
* Combines tree predictions via bootstrap aggregation (bagging) to capture non-linear feature interactions and compute feature importances.

---

## 8. Model Training

The models were trained via `src/train_model.py`. The resulting model bundle—containing the winning regressor, fitted preprocessor, and metric artifacts—is saved to `models/player_rating_model.joblib`.

---

## 9. Evaluation Metrics & Comparison

Evaluated on the hidden $20\%$ test set ($200$ players):

| Evaluation Metric | Formula | Linear Regression | Random Forest | Winner |
|---|---|:---:|:---:|:---:|
| **MAE** | $\frac{1}{n} \sum \|y - \hat{y}\|$ | **0.2941** | 0.3555 | **Linear Regression** |
| **MSE** | $\frac{1}{n} \sum (y - \hat{y})^2$ | **0.1363** | 0.1955 | **Linear Regression** |
| **RMSE** | $\sqrt{\text{MSE}}$ | **0.3692** | 0.4422 | **Linear Regression** |
| **$R^2$ Score** | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ | **0.8366 (83.66%)** | 0.7657 (76.57%) | **Linear Regression** |

### Viva Metric Explanations:
* **MAE ($\pm 0.29$ pts):** On average, the predicted rating deviates by less than $0.3$ points on a $10$-point scale.
* **RMSE ($0.37$ pts):** Measures standard deviation of residuals; confirms low variance in errors.
* **$R^2$ Score ($83.7\%$):** Over $83\%$ of the variation in player ratings across all positions is successfully explained by the feature set.

---

## 10. Visualizations

Six high-resolution plots are stored in `visualizations/`:
1. `1_rating_distribution.png`: Histogram & KDE curve confirming normal rating distribution (Mean: $7.26$, Median: $7.20$).
2. `2_goals_vs_rating.png`: Scatter plot confirming goals as primary drivers for Forwards and Wingers.
3. `3_assists_vs_rating.png`: Scatter plot demonstrating creative playmaking correlation for Midfielders.
4. `4_feature_importance.png`: Gini importance bar chart highlighting top predictive features from Random Forest.
5. `5_actual_vs_predicted.png`: Test set scatter plot showing tight clustering along the ideal $y = x$ line.
6. `6_model_comparison.png`: Bar chart contrasting MAE, RMSE, and $R^2$ across both algorithms.

---

## 11. Final Prediction System

The prediction module (`src/predict.py`) maps predictions to qualitative performance tiers:
* **$\ge 8.5$:** *Excellent* (Star / Elite Performer)
* **$7.0 - 8.4$:** *Good* (Consistent Starter)
* **$5.5 - 6.9$:** *Average* (Standard Squad Player)
* **$< 5.5$:** *Below Average* (Needs Improvement)

### Example Prediction:
```text
Input:
Age=20, Position=RW, Matches=20, Minutes=1500, Goals=8, Assists=7,
Pass_Accuracy=84%, Shots=35, Key_Passes=42, Tackles=20, Interceptions=10,
Distance_Covered=10.2 km, Sprint_Speed=28.0 km/h

Output:
>> Predicted Performance Rating: 8.0 / 10.0
>> Performance Category: Good (Quality First-Team Starter)
>> Explanation: Steady goal return (8 goals) and strong creative support (7 assists).
```

---

## 12. Multi-Profile Test Results

Automated profile tests (`tests/test_profiles.py`) confirm realistic behavior:
* **Poor-Performing Player:** Forward with $0$ goals, $0$ assists, $56\%$ pass accuracy $\rightarrow$ **$4.7 / 10.0$ (Below Average)**
* **Average-Performing Player:** Midfielder with $3$ goals, $4$ assists, $28$ tackles $\rightarrow$ **$6.8 / 10.0$ (Average)**
* **High-Performing Player:** Forward with $20$ goals, $7$ assists, $33.8\text{ km/h}$ speed $\rightarrow$ **$9.3 / 10.0$ (Excellent)**

---

## 13. How to Run the Project

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Generate Dataset
```bash
python3 dataset/generate_dataset.py
```

### 3. Run Preprocessing Checks
```bash
python3 src/preprocess.py
```

### 4. Train and Evaluate Models
```bash
python3 src/train_model.py
```

### 5. Generate Visualizations
```bash
python3 src/visualize.py
```

### 6. Run Test Profile Suite
```bash
python3 tests/test_profiles.py
```

### 7. Launch Interactive Web App
```bash
streamlit run app/app.py
```

---

## 14. Conclusion & Future Scope

### Conclusion
This project successfully demonstrates the application of Machine Learning to solve a sports analytics problem. By implementing proper preprocessing (One-Hot Encoding, StandardScaler), training both baseline and ensemble regression algorithms, and building an accessible Streamlit web application, the system achieves an $R^2$ score of $83.66\%$ and an MAE of $\pm 0.29$ rating points.

### Future Scope
1. **Position Expansion:** Incorporate Goalkeeper-specific metrics (saves, clean sheets, penalty save percentage).
2. **Spatial Tracking Data:** Integrate event data such as player heatmaps and xG (expected goals) coordinates.
3. **Opposition Adjustment:** Weigh performance metrics against the relative strength of the opposing team.
4. **Time-Series Form Modeling:** Incorporate rolling form averages to track player fatigue and momentum across game weeks.
