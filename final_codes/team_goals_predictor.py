import pandas as pd
import xgboost as xgb
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuration Constants
FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]

class TeamGoalsPredictor:
    """
    A class to predict the number of goals a team will score and concede in a match using XGBoost.
    It handles feature engineering (historical rolling averages) and model training.
    """
    def __init__(self):
        # Initialize two separate XGBoost regressors:
        self.model_scored = xgb.XGBRegressor(
            n_estimators=1000,      
            learning_rate=0.01,      
            max_depth=5,             
            subsample=0.8,           
            colsample_bytree=0.8,    
            early_stopping_rounds=50,
            n_jobs=-1                
        )
        # To predict goals conceded by the team
        self.model_conceded = xgb.XGBRegressor(
            n_estimators=1000, 
            learning_rate=0.01, 
            max_depth=5, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            early_stopping_rounds=50,
            n_jobs=-1
        )
        
    def feature_engineering(self, df):
        """
        Generates historical features for prediction.
        """
        # Sort chronologically to calculate rolling stats correctly
        df = df.sort_values(['Team_Name', 'Wk']).copy()
        
        cols_to_roll = ['Goals', 'Goals_Ag', 'xG', 'xG_Ag', 'Match_Result']
        
        # Calculate rolling averages for each team
        for col in cols_to_roll:
            # Shift by 1 row (game) so row N contains average of N-5 to N-1
            df[f'Roll_5_{col}'] = df.groupby('Team_Name')[col].transform(lambda x: x.shift(1).rolling(5).mean())
            df[f'Roll_3_{col}'] = df.groupby('Team_Name')[col].transform(lambda x: x.shift(1).rolling(3).mean())
            
        
        # OPPONENT DEFENSIVE STATS
        defensive_stats = df[['Wk', 'Team_Name', 'Roll_5_Goals_Ag', 'Roll_5_xG_Ag', 'Pre_Match_Strength']].copy()
        
        # Rename so we can merge: Team_Name -> Opponent_Name
        defensive_stats.columns = ['Wk', 'Opponent_Name', 'Opp_Roll_5_Conceded', 'Opp_Roll_5_xG_Conceded', 'Opp_Real_Strength']
        
        # Merge opponent stats back into the main dataframe based on Match Week and Opponent Name
        df = pd.merge(df, defensive_stats, on=['Wk', 'Opponent_Name'], how='left')

        return df

    def prepare_training_data(self, seasons=SEASONS, folder=FOLDER):
        """
        Loads 'teams.pkl' files for all specified seasons, concatenates them,
        applies feature engineering, and removes rows with missing values (e.g., first 5 games).
        """
        print("--- Loading and Preparing Team Data ---")
        all_dfs = []
        for season in seasons:
            path = os.path.join(folder, f"season{season}", "teams.pkl")
            if os.path.exists(path):
                d = pd.read_pickle(path)
                d['Season'] = season # Add season identifier
                all_dfs.append(d)
        
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # Apply the feature engineering logic defined above
        full_df = self.feature_engineering(full_df)
        
        # Drop the first few matches of every season where rolling averages are NaN
        full_df = full_df.dropna(subset=['Roll_5_Goals', 'Opp_Roll_5_Conceded'])
        
        return full_df

    def train(self, df):
        """
        Trains the two XGBoost models (Scored and Conceded).
        Splits data into Train/Test sets and monitors validation error to stop early if needed.
        Returns the list of features used for prediction.
        """
        print("--- Starting XGBoost Training ---")
        
        # Select predictive features (mix of Team Form + Opponent Weakness)
        features = [
            'home', 'Pre_Match_Strength', 'Opponent_Strength', 'Strength_Diff',
            'Roll_5_Goals', 'Roll_3_Goals', 'Roll_5_xG', 
            'Roll_5_Goals_Ag', 'Roll_5_xG_Ag',
            'Opp_Roll_5_Conceded', 'Opp_Roll_5_xG_Conceded'
        ]
        
        X = df[features]
        y_scored = df['Goals']
        y_conceded = df['Goals_Ag']
        
        # Split data (80% Train, 20% Test)
        X_train, X_test, y_train_s, y_test_s = train_test_split(X, y_scored, test_size=0.2, random_state=42)
        _, _, y_train_c, y_test_c = train_test_split(X, y_conceded, test_size=0.2, random_state=42)
        
        # Train Model 1: Goals Scored
        print("Training Goals Scored Model...")
        self.model_scored.fit(
            X_train, y_train_s, 
            eval_set=[(X_test, y_test_s)], 
            verbose=False
        )
        preds_s = self.model_scored.predict(X_test)
        print(f"MAE Goal Scored: {mean_absolute_error(y_test_s, preds_s):.4f}")
        
        # Train Model 2: Goals Conceded
        print("Training Goals Conceded Model...")
        self.model_conceded.fit(
            X_train, y_train_c, 
            eval_set=[(X_test, y_test_c)], 
            verbose=False
        )
        preds_c = self.model_conceded.predict(X_test)
        print(f"MAE Goal Conceded: {mean_absolute_error(y_test_c, preds_c):.4f}")
        
        return features

    def predict_and_save(self, df_full, features):
        """
        Applies the trained models to the entire dataset.
        Adds columns: 'Pred_Team_Goals', 'Pred_Team_Conceded', 'Pred_Goal_Diff'.
        """
        print("--- Generating Predictions for the full dataset ---")
        
        X = df_full[features]
        
        # Predict using trained models
        df_full['Pred_Team_Goals'] = self.model_scored.predict(X)
        df_full['Pred_Team_Conceded'] = self.model_conceded.predict(X)
        
        # Calculate expected goal difference (useful for match result prediction)
        df_full['Pred_Goal_Diff'] = df_full['Pred_Team_Goals'] - df_full['Pred_Team_Conceded']
        
        return df_full


if __name__ == "__main__":
    predictor = TeamGoalsPredictor()
    
    # Load Data
    df_with_features = predictor.prepare_training_data()
    
    # Train Models
    used_features = predictor.train(df_with_features)
    
    # Generate Predictions
    df_final = predictor.predict_and_save(df_with_features, used_features)
    
    # Check results
    print("\nPrediction Examples (Last 10 matches):")
    cols_view = ['Date', 'Team_Name', 'Opponent_Name', 'Goals', 'Pred_Team_Goals', 'Goals_Ag', 'Pred_Team_Conceded']
    print(df_final[cols_view].tail(10))