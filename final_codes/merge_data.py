from preclean_data import initial_data
from scraper_fanta import extract_fanta
import pandas as pd
from fuzzywuzzy import process, fuzz
import unicodedata
import os

# Dictionary of manual inserted replacements
replacements = {
        "Daniel Bandeira": "Dani Silva",
        "Enrico Del Prato": "Delprato",
        "Kenan Yıldız": "Yildiz",
        "Kenan Yildiz": "Yildiz",
        "Kialonda": "Gaspar K.",
        "Lautaro Gianetti": "Giannetti L.",
        "Mohamed Kaba": "Kaba",
        "Tommaso Rubino": "Rubino",
        "Oliver Nielsen": "Provstgaard",
        "Estanis": "Pedrola",
        "Andrea Belotti": "Belotti A.",
        "Ebenezer Akinsanmiro": "Akinsamiro",
        "Simeon Nwankwo": "Simy",
        "Pablo Galdames Millán": "Galdames",
        "Pepín": "Machin",                 
        "Ruan": "Tressoldi",        
        "Seydou Fini": "Fini",   
        "Triantafyllos Pasaridīs": "Pasalidis",
        "Brandon Soppy": "Soppy",
        "Emanuel Aiwum": "Aiwu",
        "Arkadiusz Reca ": "Reca",
        "Eddie Salcedo": "Salcedo Eddy",
        "Bjarki Bjarkason": "Bjarkason Bjarki Steinn",
        "Samuele Ricci": "Ricci S.",
        "Tanner Tessman": "Tessmann",
        "Nicolas Nkoulou": "N'koulou"
    }

def transliterate_name(input_str):
    """
    Cleans players' names by substituting strange letters
    """
    if pd.isna(input_str): return ""
    
    s = str(input_str).lower().strip()
    
    replacements = {
        'ð': 'd',
        'æ': 'ae',
        'ø': 'o',
        'đ': 'dj',  
        'Ł': 'L',
        'ę': 'e',
        'ł': 'l',
        'ī': 'i',
        'à': 'a'
    }
    
    for char, replacement in replacements.items():
        s = s.replace(char, replacement)

    nfkd_form = unicodedata.normalize('NFKD', s)
    s = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        
    return s.strip()


def pulisci_nome_fanta(nome_fanta):
    """
    Removes the initial letter of the name if it's followed by a point
    """
    if pd.isna(nome_fanta): return ""
    parts = str(nome_fanta).strip().split()
    
    if len(parts) > 1 and (len(parts[-1]) <= 2 or parts[-1].endswith('.')):
        return " ".join(parts[:-1]).lower()
    
    return " ".join(parts).lower()


def prepare_dataframes(df_fanta, df_stats):
    """
    Prepares the two dataframes (Fantasy Football data and Statistical data) for merging.
    It creates normalization keys based on team names and match weeks to reduce the search space later.
    """
    
    # Normalize team names to lower case and strip whitespace for consistent matching
    df_stats['team_norm'] = df_stats['Team_Name'].str.lower().str.strip()
    df_fanta['team_norm'] = df_fanta['Squadra'].str.lower().str.strip()

    # Create a unique 'match_key' combining Team and Week
    df_stats['match_key'] = df_stats['team_norm'] + "_" + df_stats['Wk'].astype(str)
    df_fanta['match_key'] = df_fanta['team_norm'] + "_" + df_fanta['id_giornata'].astype(str)

    # Clean and transliterate player names to improve fuzzy matching accuracy
    df_fanta['fanta_key'] = df_fanta['Nome'].apply(pulisci_nome_fanta).apply(transliterate_name)
    df_stats['stats_key'] = df_stats['Player'].apply(transliterate_name)

    return df_fanta, df_stats


def find_problems(df_matches, df_stats):
    """
    Analyzes the result of the matching process to identify missing players.
    If too many players are missing, it stops the execution to allow manual fixing.
    """
    # Find indices in the stats dataframe that were NOT matched
    indici_mancanti = df_stats.index.difference(df_matches['index_stats'])
    df_missing = df_stats.loc[indici_mancanti].copy()

    print(f"Total non-matched rows: {len(df_missing)}")
    print("-" * 50)

    # Get unique player names that failed matching
    nomi_unici_sbagliati = sorted(df_missing['Player'].unique())

    print(f"Players to fix ({len(nomi_unici_sbagliati)}):")
    for nome in nomi_unici_sbagliati:
        squadra = df_missing[df_missing['Player'] == nome]['Team_Name'].iloc[0]
        print(f"❌ {nome} ({squadra})")

    print("-" * 50)

    # Threshold for critical failure: if > 20 players are missing, stop to update the replacement dictionary
    if len(df_missing) > 20:
        print("❌❌ WE NEED TO FIX THE REPLACEMENTS AND RE-RUN ❌❌")
        exit()

    else:
        print("We can ignore a small amount of mismatches!!")
        print("Let's merge the two dataframes!!")


def find_matches2(df_fanta, df_stats, replacements):
    """
    Core matching logic. Tries to map players from 'df_stats' to 'df_fanta'.
    """

    matches = []
    
    # Optimization: Grouping by match_key drastically reduces complexity (O(N^2) -> O(N))
    groups = df_stats.groupby('match_key')

    for key, group_stats in groups:
        # Filter the fantasy dataframe to only include players from the current team/week
        subset_fanta = df_fanta[df_fanta['match_key'] == key]
        
        if subset_fanta.empty:
            continue
        
        # Create a map for quick lookups: Transliterated Name -> Original Name
        fanta_candidates_map = dict(zip(subset_fanta['fanta_key'], subset_fanta['Nome']))
        choices = list(fanta_candidates_map.keys())
        
        fanta_names_original_set = set(subset_fanta['Nome'].values)

        for idx, row in group_stats.iterrows():
            player_stats_name = row['Player']
            player_clean_name = row['stats_key'] 
            
            found_name = None
            score = 0
            
            # Manual Replacement Check (Highest Priority)
            if player_stats_name in replacements:
                target_name = replacements[player_stats_name]
                
                # Verify the manually mapped name actually exists in the current fantasy subset
                if target_name in fanta_names_original_set:
                    found_name = target_name
                    score = 100
            
            # Fuzzy Matching Check (If manual failed)
            if found_name is None:
                # extractOne finds the string in 'choices' most similar to 'player_clean_name'
                best_match_key, calc_score = process.extractOne(
                    player_clean_name, choices, scorer=fuzz.token_set_ratio
                )
                
                # Threshold: Only accept match if similarity is > 75%
                if calc_score > 75: 
                    found_name = fanta_candidates_map[best_match_key]
                    score = calc_score
            
            # Store the successful match
            if found_name:
                matches.append({
                    'index_stats': idx, 
                    'fanta_name': found_name, 
                    'match_score': score
                })

    df_matches = pd.DataFrame(matches)
    return df_matches


def merge_dataframes(df_stats, df_fanta, df_matches):
    """
    Performs the final merge of Stats and Fantasy data using the mapping dataframe 'df_matches'.
    It uses an INNER JOIN, meaning only fully matched rows are kept.
    """
    # Merge Stats with the bridge (df_matches) using the index
    df_merged = df_stats.merge(
        df_matches, 
        left_index=True, 
        right_on='index_stats', 
        how='inner' 
    )

    # Merge the result with Fantasy data using the matched name and match key
    final_df = df_merged.merge(
        df_fanta,
        left_on=['match_key', 'fanta_name'], 
        right_on=['match_key', 'Nome'],      
        how='inner', 
        suffixes=('_stats', '_fanta')
    )
    
    # Drop temporary columns used only for the merging logic
    cols_to_drop = [
        'stats_key', 'fanta_key', 'match_key', 'index_stats', 
        'team_norm_stats', 'team_norm_fanta', 'giornata_join',
        'match_score', 'fanta_name'
    ]
    final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns], errors='ignore')
    
    final_df = final_df.reset_index(drop=True)

    return final_df


def clean_merged(df):
    """
    Cleans the final dataframe: drops redundant columns, renames columns to English/Standard format,
    and reorders them for readability.
    """
    cols_to_drop = [
        "Player", "Wk", "Team_Name"
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    rename_map = {
        'Cod': 'Player_id',
        'home': 'Is_Home',
        'Nome': 'Name',
        'Squadra': 'Team',
        'Ruolo': 'Pos',
        'Opponent_Name': 'Opponent',
        'id_giornata': 'Week_id',
        'Gls': 'Player_Goals',
        'Sh': 'Shots',          
        'CrdY': 'CardY',        
        'CrdR': 'CardR',       
        'xG_x': 'Player_xG',
        'npxG': 'Player_npxG',
        'xAG': 'Player_xAST',   
        'Goals': 'Team_Goals',  
        'Goals_Ag': 'Opp_Goals',
        'xG': 'Team_xG',        
        'xG_Ag': 'Opp_xG',
        'Au': 'Autogoal',
        'Rigori_Sbagliati': 'Missed_Pen',
        'Qt.I': 'Value',
        'Att_x.1': 'Cross'
    }
    df = df.rename(columns=rename_map)

    # Enforce a specific column order
    final_order = [
        'Player_id', 'Name', 'Team', 'Pos', 'Age', 'Value', 'Opponent', 'Week_id', 
        'Min', 'Player_Goals', 'Ast', 'Autogoal', 'Missed_Pen', 'PK', 'PKatt', 'Shots', 'SoT', 
        'CardY', 'CardR', 'Touches', 'Tkl', 'Int', 'Blocks', 
        'Player_xG', 'Player_npxG', 'Player_xAST', 
        'SCA', 'GCA', 'Cmp_x', 'Att_x', 'Cmp%_x', 'PrgP', 'Carries', 'PrgC', 
        'Cross', 'Succ', 'game_id',
        'Voto_Fantacalcio', 'Voto_Statistico', 'Voto_Italia', 'sv', 'Fantavoto'
    ]
    df = df[final_order]

    return df


def return_final_dataframe(season, folder, league="Serie-A"):
    """
    Orchestrator function. Loads raw data, runs the matching pipeline, cleans results,
    and returns the final polished dataframe ready for ML.
    """
    # Load Initial Data
    df_teams, df_stats = initial_data(season, folder, league)
    df_fanta = pd.read_pickle(os.path.join(folder, "Fantacalcio.pkl"))

    # Prepare Keys
    df_fanta, df_stats = prepare_dataframes(df_fanta, df_stats)

    # Find Matches
    print("Find Matches between the two dataframes...")
    df_matches = find_matches2(df_fanta, df_stats, replacements)

    # Check for Errors
    find_problems(df_matches, df_stats)

    # Merge and Clean
    df_final = merge_dataframes(df_stats, df_fanta, df_matches)
    print(f"✅ Merge completed! Final Rows: {len(df_final)}")
    df_final = clean_merged(df_final)
    
    return df_final