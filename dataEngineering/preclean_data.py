import pandas as pd
import numpy as np
import os


def preclean_players(season, folder, league = "Serie-A"):
    cols_to_drop = [
    'SoTA',
    'GA',
    'Saves', 
    'Save%', 
    'PSxG',
    'Cmp_y',
    'Att_y',
    'Cmp%_y', 
    'Att (GK)',
    'Thr',
    'Launch%', 
    'AvgLen', 
    'Att_y.1',
    'Launch%.1', 
    'AvgLen.1', 
    'Opp', 
    'Stp',
    'Stp%', 
    '#OPA',
    'AvgDist'
    ]

    player_file = os.path.join(folder, f'{league.lower()}_{season.lower()}_player_data.csv')
    try:
        df = pd.read_csv(player_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df_final = df[df['Pos'] != 'GK']

    return df_final


def merge_data_teams(season, folder, league = "Serie-A"):
    fixture_file = os.path.join(folder, f'{league.lower()}_{season.lower()}_fixture_data.csv')
    player_file = os.path.join(folder, f'{league.lower()}_{season.lower()}_player_data.csv')
    
    print(f"Loading {fixture_file} and {player_file}...")
    
    try:
        df_fixtures = pd.read_csv(fixture_file)
        df_fixtures = df_fixtures.dropna(subset=['Wk'])
        df_players = preclean_players(season, folder, league)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    df_fixtures["game_id"] = df_fixtures.index

    full_data = pd.merge(df_players, df_fixtures, on='game_id', how='left')
    
    full_data['Team_Name'] = np.where(full_data['home'] == 1, full_data['Home'], full_data['Away'])
    
    full_data['Opponent_Name'] = np.where(full_data['home'] == 1, full_data['Away'], full_data['Home'])

    cols_to_move = ['Date', 'Team_Name', 'Opponent_Name', 'home', 'Result']
    
    cols = [c for c in cols_to_move if c in full_data.columns] + [c for c in full_data.columns if c not in cols_to_move]
    full_data = full_data[cols]

    full_data['Team_Name'] = np.where(full_data['Team_Name'] == "Hellas Verona", "Verona", full_data['Team_Name'])
    full_data['Opponent_Name'] = np.where(full_data['Opponent_Name'] == "Hellas Verona", "Verona", full_data['Opponent_Name'])

    output_filename = os.path.join(folder, f'{league.lower()}_{season.lower()}_combined_data.csv')
    full_data.to_csv(output_filename, index=False)
    
    print(f"Success! Combined data saved to: {output_filename}")
    print(f"Total rows: {len(full_data)}")

    df_team = full_data[full_data['#'].isna()]
    df_players = full_data[full_data['#'].notna()]

    return df_team, df_players


def clean_team(df):
    cols_to_drop = [
        'Player',
        '#',
        'Pos',
        'Nation',
        'Age',
        'Min',
        'Home',
        'Away',
        'Day',
        'Time',
        'season',
        'Ast',
        'Touches',
        'Tkl',
        'Int',
        'Blocks',
        'npxG',
        'xAG',
        'PK',
        'PKatt',
        'Cmp_x',
        'Att_x',
        'Carries',
        'xG_x'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df = df.dropna()
    df['Wk'] = df['Wk'].astype(int)
    df = df.rename(columns = {"xG_y":"xG_home", "xG.1":"xG_away"})

    split_data = df['Score'].str.split('–', expand=True).astype(int)
    df['home_goals_temp'] = split_data[0]
    df['away_goals_temp'] = split_data[1]

    df['Goals'] = np.where(df['home'] == 1, df['home_goals_temp'], df['away_goals_temp'])
    df['Goals_Ag'] = np.where(df['home'] == 1, df['away_goals_temp'], df['home_goals_temp'])

    df['xG'] = np.where(df['home'] == 1, df['xG_home'], df['xG_away'])
    df['xG_Ag'] = np.where(df['home'] == 1, df['xG_away'], df['xG_home'])

    df = df.drop(columns=['Score', 'home_goals_temp', 'away_goals_temp', 'Gls', 'xG_away', 'xG_home'])

    df['Date'] = pd.to_datetime(df['Date'])
    start_date = '2020-05-01'
    end_date = '2020-10-31'

    df['is_empty'] = ((df['Date'] >= start_date) & (df['Date'] <= end_date)).astype(int)
    df['Match_Goal_Diff'] = df['Goals'] - df['Goals_Ag']
    df['xG_Diff'] = df['xG'] - df['xG_Ag']

    conditions = [
        (df['Match_Goal_Diff'] > 0),
        (df['Match_Goal_Diff'] == 0),
        (df['Match_Goal_Diff'] < 0)
    ]
    df['Match_Result'] = np.select(conditions, [1, 0, -1])

    return df


def add_league_position(df):
    df_calc = df.copy()

    conditions = [
        (df_calc['Goals'] > df_calc['Goals_Ag']), 
        (df_calc['Goals'] == df_calc['Goals_Ag']) 
    ]
    choices = [3, 1]
    
    df_calc['Match_Points'] = np.select(conditions, choices, default=0)
    
    df_calc['Match_GD'] = df_calc['Goals'] - df_calc['Goals_Ag']

    df_calc = df_calc.sort_values(['Team_Name', 'Wk'])
    
    df_calc['Cum_Points'] = df_calc.groupby('Team_Name')['Match_Points'].cumsum()
    df_calc['Cum_GD'] = df_calc.groupby('Team_Name')['Match_GD'].cumsum()
    df_calc['Cum_Goals'] = df_calc.groupby('Team_Name')['Goals'].cumsum()

    df_calc = df_calc.sort_values(
        ['Wk', 'Cum_Points', 'Cum_GD', 'Cum_Goals'], 
        ascending=[True, False, False, False]
    )

    df_calc['League_Pos'] = df_calc.groupby('Wk').cumcount() + 1

    final_ranking = df_calc[['game_id', 'Team_Name', 'League_Pos']]
    
    df_out = df.merge(final_ranking, on=['game_id', 'Team_Name'], how='left')
    
    return df_out


def clean_players(df):
    cols_to_drop = [
        '#',
        'Pos',
        'Nation',
        'Home',
        'Away',
        'Day',
        'Time',
        'season'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df = df.dropna()

    df['Age'] = df['Age'].astype(str).str[:2].astype(int)
    df['Wk'] = df['Wk'].astype(int)

    df = df.rename(columns = {"xG_y":"xG_home", "xG.1":"xG_away"})

    split_data = df['Score'].str.split('–', expand=True).astype(int)
    df['home_goals_temp'] = split_data[0]
    df['away_goals_temp'] = split_data[1]

    df['Goals'] = np.where(df['home'] == 1, df['home_goals_temp'], df['away_goals_temp'])
    df['Goals_Ag'] = np.where(df['home'] == 1, df['away_goals_temp'], df['home_goals_temp'])

    df['xG'] = np.where(df['home'] == 1, df['xG_home'], df['xG_away'])
    df['xG_Ag'] = np.where(df['home'] == 1, df['xG_away'], df['xG_home'])

    df = df.drop(columns=['Score', 'home_goals_temp', 'away_goals_temp', 'xG_away', 'xG_home'])

    return df


def calculate_team_strength(df):
    df = df.sort_values(['Team_Name', 'Wk']).copy()
    
    conditions = [
        (df['Goals'] > df['Goals_Ag']), 
        (df['Goals'] == df['Goals_Ag'])
    ]
    df['Raw_Points'] = np.select(conditions, [3, 1], default=0)
    
    difficulty_multiplier = np.where(df['home'] == 0, 1.2, 1.0)
    df['Adjusted_Points'] = df['Raw_Points'] * difficulty_multiplier
    
    df['xG_Ratio'] = df['xG'] / (df['xG'] + df['xG_Ag'] + 0.01)

    
    df['Form_Points_EMA'] = df.groupby('Team_Name')['Adjusted_Points'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
    df['Form_xG_EMA'] = df.groupby('Team_Name')['xG_Ratio'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
    
    df['Form_Score'] = ((df['Form_Points_EMA'] / 3.6) * 0.60 + (df['Form_xG_EMA'] * 0.40)) * 100
    
    df['Season_PPG_Current'] = df.groupby('Team_Name')['Raw_Points'].transform(lambda x: x.expanding().mean())
    
    df['Pre_Match_PPG'] = df.groupby('Team_Name')['Season_PPG_Current'].shift(1)
    
    df['Pre_Match_PPG'] = df['Pre_Match_PPG'].fillna(1.35)
    
    df['Status_Score'] = (df['Pre_Match_PPG'] / 3.0) * 100
    
    max_wk = df['Wk'].max()
    df['Season_Progress'] = df['Wk'] / max_wk
    
    df['Weight_Status'] = (df['Season_Progress'] * 0.5).clip(0.1, 0.5)
    df['Weight_Form'] = 1 - df['Weight_Status']
    
    df['Final_Strength_Raw'] = (df['Form_Score'] * df['Weight_Form']) + \
                               (df['Status_Score'] * df['Weight_Status'])
    
    df['Pre_Match_Strength'] = df.groupby('Team_Name')['Final_Strength_Raw'].shift(1)
    
    df['Pre_Match_Strength'] = df['Pre_Match_Strength'].fillna(50.0)

    cols_to_drop = ['Raw_Points', 'Adjusted_Points', 'xG_Ratio', 'Form_Points_EMA', 'Form_xG_EMA', 
                    'Form_Score', 'Season_PPG_Current', 'Pre_Match_PPG', 'Status_Score', 
                    'Season_Progress', 'Weight_Status', 'Weight_Form', 'Final_Strength_Raw']
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df


def add_opponent_strength_features(df):
    opponent_lookup = df[['game_id', 'Team_Name', 'Pre_Match_Strength']].copy()
    
    opponent_lookup = opponent_lookup.rename(columns={
        'Team_Name': 'Opponent_Name',
        'Pre_Match_Strength': 'Opponent_Strength'
    })
    
    df_enriched = df.merge(opponent_lookup, on=['game_id', 'Opponent_Name'], how='left')
    
    df_enriched['Strength_Diff'] = df_enriched['Pre_Match_Strength'] - df_enriched['Opponent_Strength']
    
    return df_enriched


def initial_data(season, folder, league = "Serie-A"):
    df_teams, df_players = merge_data_teams(season, folder, league)

    df_teams = clean_team(df_teams)
    df_teams = add_league_position(df_teams)
    df_teams = calculate_team_strength(df_teams)
    df_teams = add_opponent_strength_features(df_teams)
    
    df_players = clean_players(df_players)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    #print("\nInformation about the teams!!\n")
    #print(df_teams.head(10))

    #print("\nInformation about the players!!\n")
    #print(df_players.head(20))

    return df_teams, df_players