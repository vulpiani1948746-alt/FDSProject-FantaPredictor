import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import merge_players_with_teams
from sklearn.preprocessing import StandardScaler


FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Put your folder that will contain all the data
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 


class RobustMinMaxScaler:
    def __init__(self, feature_range=(0, 1), clip_quantiles=(0.01, 0.99)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.clip_quantiles = clip_quantiles
        self.bounds = {}
        self.features = []

    def fit(self, df, columns):
        self.features = columns
        for col in columns:
            lower = df[col].quantile(self.clip_quantiles[0])
            upper = df[col].quantile(self.clip_quantiles[1])
            self.bounds[col] = (lower, upper)
        
        df_clipped = df[columns].copy()
        for col in columns:
            lower, upper = self.bounds[col]
            df_clipped[col] = df_clipped[col].clip(lower, upper)
            
        self.scaler.fit(df_clipped)
        print("Scaler Robusto addestrato (Outlier gestiti).")

    def transform(self, df):
        df_transformed = df.copy()
        
        for col in self.features:
            if col in self.bounds:
                lower, upper = self.bounds[col]
                df_transformed[col] = df_transformed[col].clip(lower, upper)
        
        df_transformed[self.features] = self.scaler.transform(df_transformed[self.features])
        
        return df_transformed


def create_scaler(player_cols, team_past_cols, team_future_cols, scaler_type, seasons = SEASONS, folder = FOLDER):
    list_players, list_teams = [], []

    for season in seasons[:-2]:
        df_players = pd.read_pickle(os.path.join(folder, f"season{season}", "FinalDF_player.pkl"))
        df_teams = pd.read_pickle(os.path.join(folder, f"season{season}", "teams.pkl"))

        list_players.append(df_players)
        list_teams.append(df_teams)

    training_frames = []

    for i in range(5):
        df_merged = merge_players_with_teams(list_players[i], list_teams[i])
    
        training_frames.append(df_merged)

    full_training_raw = pd.concat(training_frames, ignore_index=True)

    print(f"Dimensioni Mega-DataFrame Training: {full_training_raw.shape}")

    cols_to_scale = player_cols + team_past_cols + team_future_cols

    if scaler_type == 'robust':
        scaler = RobustMinMaxScaler(clip_quantiles=(0.01, 0.99))
        scaler.fit(full_training_raw, cols_to_scale)
        
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(full_training_raw[cols_to_scale])
        
    elif scaler_type == 'standard':
        scaler = StandardScaler()
        scaler.fit(full_training_raw[cols_to_scale])
        
    else:
        raise ValueError("Scaler non supportato. Usa 'robust', 'minmax' o 'standard'")

    robust_scaler = RobustMinMaxScaler(clip_quantiles=(0.01, 0.99))

    robust_scaler.fit(full_training_raw, cols_to_scale)

    return robust_scaler
    
