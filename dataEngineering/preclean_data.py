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

    output_filename = f'{folder}{league.lower()}_{season.lower()}_combined_data.csv'
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

    return df


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


def initial_data(season, folder, league = "Serie-A"):
    df_teams, df_players = merge_data_teams(season, folder, league)

    df_teams = clean_team(df_teams)
    df_players = clean_players(df_players)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    #print("\nInformation about the teams!!\n")
    #print(df_teams.head(10))

    #print("\nInformation about the players!!\n")
    #print(df_players.head(20))

    return df_teams, df_players