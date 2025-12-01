import pandas as pd
import numpy as np

def merge_players_with_teams(df_players, df_teams):
    rename_map = {
        'Goals': 'Team_Goals',  
        'Goals_Ag': 'Opp_Goals',
        'xG': 'Team_xG',        
        'xG_Ag': 'Opp_xG',
    }
    df_teams = df_teams.rename(columns=rename_map)
    
    cols_to_use = [
        'game_id',                  # Chiave fondamentale per il merge
        'Team_Name',                # Chiave fondamentale per il merge
        'Pre_Match_Strength',
        'Opponent_Strength',        # La forza dell'avversario
        'Strength_Diff',            # La differenza di forza
        'is_empty',              # Feature stadio vuoto
        'Team_Goals',
        'Opp_Goals',
        'Match_Goal_Diff',
        'Team_xG',
        'Opp_xG',
        'xG_Diff',
        'Match_Result',
        'home',
        'Pred_Team_Goals',
        'Pred_Team_Conceded',
        'Pred_Goal_Diff'
    ]
    
    df_teams_subset = df_teams[cols_to_use].copy()
    
    df_merged = pd.merge(
        df_players,
        df_teams_subset,
        left_on=['game_id', 'Team'],      # Chiavi in df_players
        right_on=['game_id', 'Team_Name'],# Chiavi in df_teams
        how='left'                        # Manteniamo tutti i giocatori, attacchiamo le info squadra
    )
    
    df_merged = df_merged.drop(columns=['Team_Name'])

    return df_merged

def feature_engineering_player(df):
    df = df.sort_values(['Player_id', 'game_id']).copy()    
    
    df['Player_xG_Overperf'] = df['Player_Goals'] - df['Player_xG']
    
    df['Shot_Accuracy'] = np.where(df['Shots'] > 0, df['SoT'] / df['Shots'], 0.0)

    df = pd.get_dummies(df, columns=['Pos'], prefix='Role', dtype=int)

    df["Voto_Diff"] = df["Voto_Fantacalcio"] - ((df["Voto_Statistico"] + df["Voto_Italia"])/2.0)

    cols_to_avg = ['Voto_Fantacalcio', 'Fantavoto', 'Voto_Diff']

    for col in cols_to_avg:
        col_name = f'Avg_{col}'
        
        df[col_name] = df.groupby('Player_id')[col].transform(
            lambda x: x.expanding().mean().shift(1)
        )

    return df
