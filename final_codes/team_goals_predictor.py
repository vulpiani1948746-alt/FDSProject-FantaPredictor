import pandas as pd
import xgboost as xgb
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]

class TeamGoalsPredictor:
    def __init__(self):
        self.model_scored = xgb.XGBRegressor(
            n_estimators=1000, 
            learning_rate=0.01, 
            max_depth=5, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            early_stopping_rounds=50,
            n_jobs=-1
        )
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
        Crea le feature basate sullo storico (ultime N partite).
        È fondamentale usare .shift() per non usare i dati della partita futura per prevederla!
        """
        df = df.sort_values(['Team_Name', 'Wk']).copy()
        
        cols_to_roll = ['Goals', 'Goals_Ag', 'xG', 'xG_Ag', 'Match_Result']
        
        for col in cols_to_roll:
            df[f'Roll_5_{col}'] = df.groupby('Team_Name')[col].transform(lambda x: x.shift(1).rolling(5).mean())
            df[f'Roll_3_{col}'] = df.groupby('Team_Name')[col].transform(lambda x: x.shift(1).rolling(3).mean())
            
        
        defensive_stats = df[['Wk', 'Team_Name', 'Roll_5_Goals_Ag', 'Roll_5_xG_Ag', 'Pre_Match_Strength']].copy()
        defensive_stats.columns = ['Wk', 'Opponent_Name', 'Opp_Roll_5_Conceded', 'Opp_Roll_5_xG_Conceded', 'Opp_Real_Strength']
        
        df = pd.merge(df, defensive_stats, on=['Wk', 'Opponent_Name'], how='left')

        return df

    def prepare_training_data(self, seasons=SEASONS, folder=FOLDER):
        print("--- Caricamento e Preparazione Dati Squadre ---")
        all_dfs = []
        for season in seasons:
            path = os.path.join(folder, f"season{season}", "teams.pkl")
            if os.path.exists(path):
                d = pd.read_pickle(path)
                d['Season'] = season
                all_dfs.append(d)
        
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        full_df = self.feature_engineering(full_df)
        
        full_df = full_df.dropna(subset=['Roll_5_Goals', 'Opp_Roll_5_Conceded'])
        
        return full_df

    def train(self, df):
        print("--- Inizio Training Modelli XGBoost ---")
        
        features = [
            'home', 'Pre_Match_Strength', 'Opponent_Strength', 'Strength_Diff',
            'Roll_5_Goals', 'Roll_3_Goals', 'Roll_5_xG', 
            'Roll_5_Goals_Ag', 'Roll_5_xG_Ag',
            'Opp_Roll_5_Conceded', 'Opp_Roll_5_xG_Conceded'
        ]
        
        targets = ['Goals', 'Goals_Ag']
        
        X = df[features]
        y_scored = df['Goals']
        y_conceded = df['Goals_Ag']
        
        X_train, X_test, y_train_s, y_test_s = train_test_split(X, y_scored, test_size=0.2, random_state=42)
        _, _, y_train_c, y_test_c = train_test_split(X, y_conceded, test_size=0.2, random_state=42)
        
        print("Training Modello Goal Fatti...")
        self.model_scored.fit(
            X_train, y_train_s, 
            eval_set=[(X_test, y_test_s)], 
            verbose=False
        )
        preds_s = self.model_scored.predict(X_test)
        print(f"MAE Goal Scored: {mean_absolute_error(y_test_s, preds_s):.4f}")
        
        print("Training Modello Goal Subiti...")
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
        Aggiunge le colonne 'Pred_Team_Goals' e 'Pred_Team_Conceded' al dataframe originale
        """
        print("--- Generazione Previsioni su tutto il dataset ---")
        
        X = df_full[features]
        
        df_full['Pred_Team_Goals'] = self.model_scored.predict(X)
        df_full['Pred_Team_Conceded'] = self.model_conceded.predict(X)
        
        df_full['Pred_Goal_Diff'] = df_full['Pred_Team_Goals'] - df_full['Pred_Team_Conceded']
        
        return df_full


if __name__ == "__main__":
    predictor = TeamGoalsPredictor()
    
    df_with_features = predictor.prepare_training_data()
    
    used_features = predictor.train(df_with_features)
    
    df_final = predictor.predict_and_save(df_with_features, used_features)
    
    print("\nEsempio Previsioni (Ultime 10 partite):")
    cols_view = ['Date', 'Team_Name', 'Opponent_Name', 'Goals', 'Pred_Team_Goals', 'Goals_Ag', 'Pred_Team_Conceded']
    print(df_final[cols_view].tail(10))
    
    # df_final.to_pickle(os.path.join(FOLDER, "Teams_With_Predictions.pkl"))