"""
Script to generate notebooks/analysis.ipynb
"""

import json
import os

notebook_data = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Football Player Performance Prediction & Rating System\n",
    "### AIML Mini Project — Experiment 12\n",
    "**Objective:** Predict a player's seasonal performance rating (scale 1.0 to 10.0) from match statistics using Machine Learning regression."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Libraries & Load Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv('../dataset/football_players.csv')\n",
    "print(f'Dataset Shape: {df.shape}')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Exploratory Data Analysis & Integrity Checks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Missing values and duplicates check\n",
    "print('Missing values:\\n', df.isnull().sum())\n",
    "print('\\nDuplicate rows:', df.duplicated().sum())\n",
    "\n",
    "# Summary statistics of target variable\n",
    "print('\\nTarget Variable Summary:')\n",
    "print(df['Performance_Rating'].describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing & Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "\n",
    "num_cols = ['Age', 'Matches_Played', 'Minutes_Played', 'Goals', 'Assists', 'Pass_Accuracy', \n",
    "            'Shots', 'Key_Passes', 'Tackles', 'Interceptions', 'Distance_Covered', 'Sprint_Speed']\n",
    "cat_cols = ['Position']\n",
    "target_col = 'Performance_Rating'\n",
    "\n",
    "X = df[num_cols + cat_cols]\n",
    "y = df[target_col]\n",
    "\n",
    "X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "preprocessor = ColumnTransformer([\n",
    "    ('num', StandardScaler(), num_cols),\n",
    "    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)\n",
    "])\n",
    "\n",
    "X_train = preprocessor.fit_transform(X_train_raw)\n",
    "X_test = preprocessor.transform(X_test_raw)\n",
    "\n",
    "print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Model Training & Comparison (Linear Regression vs Random Forest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "\n",
    "# 1. Linear Regression\n",
    "lr = LinearRegression()\n",
    "lr.fit(X_train, y_train)\n",
    "lr_preds = lr.predict(X_test)\n",
    "\n",
    "# 2. Random Forest Regressor\n",
    "rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)\n",
    "rf.fit(X_train, y_train)\n",
    "rf_preds = rf.predict(X_test)\n",
    "\n",
    "def get_metrics(y_true, y_pred):\n",
    "    mae = mean_absolute_error(y_true, y_pred)\n",
    "    mse = mean_squared_error(y_true, y_pred)\n",
    "    rmse = np.sqrt(mse)\n",
    "    r2 = r2_score(y_true, y_pred)\n",
    "    return mae, mse, rmse, r2\n",
    "\n",
    "lr_mae, lr_mse, lr_rmse, lr_r2 = get_metrics(y_test, lr_preds)\n",
    "rf_mae, rf_mse, rf_rmse, rf_r2 = get_metrics(y_test, rf_preds)\n",
    "\n",
    "results_df = pd.DataFrame({\n",
    "    'Metric': ['MAE', 'MSE', 'RMSE', 'R2 Score'],\n",
    "    'Linear Regression': [lr_mae, lr_mse, lr_rmse, lr_r2],\n",
    "    'Random Forest Regressor': [rf_mae, rf_mse, rf_rmse, rf_r2]\n",
    "})\n",
    "results_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visualizing Actual vs Predicted Ratings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(7, 5))\n",
    "plt.scatter(y_test, lr_preds, color='#1f77b4', alpha=0.7, edgecolors='k', label='Players (N=200)')\n",
    "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect (y = x)')\n",
    "plt.title('Actual vs Predicted Ratings on Test Set')\n",
    "plt.xlabel('Actual Performance Rating')\n",
    "plt.ylabel('Predicted Performance Rating')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/analysis.ipynb", "w") as f:
    json.dump(notebook_data, f, indent=2)

print("Created notebooks/analysis.ipynb successfully!")
