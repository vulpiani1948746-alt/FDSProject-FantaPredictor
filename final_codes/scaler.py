import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from feature_engineering import merge_players_with_teams


FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Path to data directory
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 


class RobustMinMaxScaler:
    """
    A custom scaler that combines outlier clipping with MinMax scaling.
    """
    def __init__(self, feature_range=(0, 1), clip_quantiles=(0.01, 0.99)):
        # Initialize the underlying standard MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=feature_range)
        # Define percentile thresholds for clipping outliers (e.g., bottom 1% and top 1%)
        self.clip_quantiles = clip_quantiles
        # Dictionaries to store the learned boundaries (min/max) for each column
        self.bounds = {}
        self.features = []

    def fit(self, df, columns):
        """
        Calculates the clipping boundaries (quantiles) and fits the MinMaxScaler.
        """
        self.features = columns
        
        # Learn Boundaries
        for col in columns:
            lower = df[col].quantile(self.clip_quantiles[0])
            upper = df[col].quantile(self.clip_quantiles[1])
            self.bounds[col] = (lower, upper)
        
        # Create a temporary clipped dataframe
        df_clipped = df[columns].copy()
        for col in columns:
            lower, upper = self.bounds[col]
            df_clipped[col] = df_clipped[col].clip(lower, upper)
            
        # Fit the standard scaler on the clean, clipped data
        self.scaler.fit(df_clipped)
        print("Robust Scaler trained (Outliers handled).")

    def transform(self, df):
        """
        Applies the learned clipping and scaling to new data.
        """
        df_transformed = df.copy()
        
        # Apply Clipping using learned bounds (prevents new outliers from breaking the scale)
        for col in self.features:
            if col in self.bounds:
                lower, upper = self.bounds[col]
                df_transformed[col] = df_transformed[col].clip(lower, upper)
        
        # Apply Standard Scaling (0-1 normalization)
        df_transformed[self.features] = self.scaler.transform(df_transformed[self.features])
        
        return df_transformed


def create_scaler(player_cols, team_past_cols, team_future_cols, scaler_type, seasons=SEASONS, folder=FOLDER):
    """
    Loads historical training data, merges player and team stats, and fits a global scaler.
    """
    list_players, list_teams = [], []

    # LOAD TRAINING DATA
    for season in seasons[:-2]:
        df_players = pd.read_pickle(os.path.join(folder, f"season{season}", "FinalDF_player.pkl"))
        df_teams = pd.read_pickle(os.path.join(folder, f"season{season}", "teams.pkl"))

        list_players.append(df_players)
        list_teams.append(df_teams)

    # MERGE DATA
    training_frames = []
    for i in range(5):
        df_merged = merge_players_with_teams(list_players[i], list_teams[i])
        training_frames.append(df_merged)

    # Concatenate all historical seasons into one massive dataframe
    full_training_raw = pd.concat(training_frames, ignore_index=True)

    print(f"Mega-DataFrame Training Size: {full_training_raw.shape}")

    cols_to_scale = player_cols + team_past_cols + team_future_cols

    # INITIALIZE AND FIT CHOSEN SCALER
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
        raise ValueError("Scaler not supported. Use 'robust', 'minmax' or 'standard'")

    robust_scaler = RobustMinMaxScaler(clip_quantiles=(0.01, 0.99))
    robust_scaler.fit(full_training_raw, cols_to_scale)

    return robust_scaler