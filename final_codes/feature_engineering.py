import pandas as pd
import numpy as np

def merge_players_with_teams(df_players, df_teams):
    """
    Merges player stats with team match stats (e.g., result, opponent strength).
    """
    # Rename generic team columns to avoid confusion with player stats
    rename_map = {
        'Goals': 'Team_Goals',  
        'Goals_Ag': 'Opp_Goals',
        'xG': 'Team_xG',        
        'xG_Ag': 'Opp_xG',
    }
    df_teams = df_teams.rename(columns=rename_map)
    
    # Select only relevant team context features for the merge
    cols_to_use = [
        'game_id', 'Team_Name', 'Pre_Match_Strength', 'Opponent_Strength', 
        'Strength_Diff', 'is_empty', 'Team_Goals', 'Opp_Goals', 
        'Match_Goal_Diff', 'Team_xG', 'Opp_xG', 'xG_Diff', 
        'Match_Result', 'home', 'Pred_Team_Goals', 
        'Pred_Team_Conceded', 'Pred_Goal_Diff'
    ]
    
    df_teams_subset = df_teams[cols_to_use].copy()
    
    # Left merge to keep all player records, matching on Game ID and Team Name
    df_merged = pd.merge(
        df_players,
        df_teams_subset,
        left_on=['game_id', 'Team'],      
        right_on=['game_id', 'Team_Name'],
        how='left'                        
    )
    
    return df_merged.drop(columns=['Team_Name'])


def feature_engineering_player(df):
    """
    Creates derived features: xG performance, shot accuracy, and historical averages.
    """
    # Sort by player and time to ensure rolling calculations are correct
    df = df.sort_values(['Player_id', 'game_id']).copy()    
    
    # Calculate performance metrics
    df['Player_xG_Overperf'] = df['Player_Goals'] - df['Player_xG']
    df['Shot_Accuracy'] = np.where(df['Shots'] > 0, df['SoT'] / df['Shots'], 0.0)

    # One-hot encode player positions (e.g., Role_D, Role_C)
    df = pd.get_dummies(df, columns=['Pos'], prefix='Role', dtype=int)

    # Compare editorial rating (Fantacalcio) vs statistical/paper ratings
    df["Voto_Diff"] = df["Voto_Fantacalcio"] - ((df["Voto_Statistico"] + df["Voto_Italia"])/2.0)

    # Calculate expanding averages (shifted by 1 to prevent data leakage)
    cols_to_avg = ['Voto_Fantacalcio', 'Fantavoto', 'Voto_Diff']

    for col in cols_to_avg:
        col_name = f'Avg_{col}'
        df[col_name] = df.groupby('Player_id')[col].transform(
            lambda x: x.expanding().mean().shift(1)
        )

    return df
