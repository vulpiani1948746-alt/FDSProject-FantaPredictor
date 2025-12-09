from preclean_data import initial_data
from scraper_fanta import fanta_scraper
from scraper_fanta import extract_fanta
from merge_data import return_final_dataframe
from feature_engineering import merge_players_with_teams, feature_engineering_player
from feature_selection import drop_useless_features
from team_goals_predictor import TeamGoalsPredictor 
import pandas as pd
import os
import time

###### CHANGE THESE VALUES!!!
FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 

def split_and_save_roles(df, season_folder):
    """
    Divides the main dataframe in specific one for the roles and save them
    """
    
    roles_map = {
        #'P': 'FinalDF_Role_P.pkl', 
        'D': 'FinalDF_Role_D.pkl',  
        'C': 'FinalDF_Role_C.pkl',  
        'A': 'FinalDF_Role_A.pkl'   
    }
    
    saved_files = []

    # Checks if the column of that role is 1 and saves it in the new dataframe
    for role_code, filename in roles_map.items():
        col_name = f"Role_{role_code}"
        
        if col_name in df.columns:
            df_role = df[df[col_name] == 1].copy()
            
            if not df_role.empty:
                save_path = os.path.join(season_folder, filename)
                df_role.to_pickle(save_path)
                saved_files.append(filename)

        elif 'Role' in df.columns:
            df_role = df[df['Role'] == role_code].copy()
            if not df_role.empty:
                save_path = os.path.join(season_folder, filename)
                df_role.to_pickle(save_path)
                saved_files.append(filename)

    print(f"Role dataframes created: {', '.join(saved_files)}")


def create_dfs():
    """
    Main orchestration function to process data for all defined seasons.
    It iterates through each season, initializes data, predicts team goals,
    performs feature engineering, splits data by player roles, and merges final datasets.
    """
    
    # Initialize the team goals predictor engine
    predictor = TeamGoalsPredictor()

    # Loop through each season defined in the SEASONS list
    for season in SEASONS:
        print(f"\n\n\nLet's do season {season}")
        print("-" * 50)
        print("\n\n")

        final_folder = os.path.join(FOLDER, f"season{season}")
        os.makedirs(final_folder, exist_ok=True)
        
        df_teams, df_stats = initial_data(season, final_folder)
        print(f"LENGTH OF PLAYERS DATAFRAME: {len(df_stats.columns)}")
        print(f"LENGTH OF TEAMS DATAFRAME: {len(df_teams.columns)}")

        df_teams.to_pickle(os.path.join(final_folder, "teams.pkl"))
        df_stats.to_pickle(os.path.join(final_folder, "stats.pkl"))

        print(f"Calculating Team Goal predictions for {season}...")
        try:
            current_idx = SEASONS.index(season)
            seasons_so_far = SEASONS[:current_idx+1]
            
            # Prepare data for the predictor using historical data up to now
            df_aug = predictor.prepare_training_data(seasons=seasons_so_far, folder=FOLDER)
            
            # Ensure there is enough data to train a meaningful model
            if len(df_aug) > 50:
                used_feats = predictor.train(df_aug)
                
                # Predict goals/conceded goals and get a dataframe with predictions
                df_with_preds = predictor.predict_and_save(df_aug, used_feats)
                
                df_preds_current = df_with_preds[df_with_preds['Season'] == season]
                
                df_preds_current = df_preds_current.drop_duplicates(subset=['Team_Name', 'Wk'])
                cols_to_merge = ['Team_Name', 'Wk', 'Pred_Team_Goals', 'Pred_Team_Conceded', 'Pred_Goal_Diff']
                
                if 'Pred_Team_Goals' in df_preds_current.columns:
                    # Merge predictions back into the main team dataframe
                    df_teams = df_teams.merge(
                        df_preds_current[cols_to_merge], 
                        on=['Team_Name', 'Wk'], 
                        how='left'
                    )
                    print(f"Predictions integrated for {season}!")
            else:
                print(f"Not enough data to train model (normal for {season}). Skipping.")

        except Exception as e:
            print(f"Error calculating predictions (normal if missing historical data): {e}")

        # Fill missing prediction columns with 0.0 if the prediction step failed or was skipped
        for col in ['Pred_Team_Goals', 'Pred_Team_Conceded', 'Pred_Goal_Diff']:
            if col not in df_teams.columns:
                df_teams[col] = 0.0
        
        # Save the updated team dataframe with predictions
        df_teams.to_pickle(os.path.join(final_folder, "teams.pkl"))

        print("Let's merge the two DataFrames...")
        # Combine initial team and player data
        df_final = return_final_dataframe(season, final_folder)
        print(f"LENGTH OF MERGED DATAFRAME: {len(df_final.columns)}")
        print("-" * 50)

        # Generate advanced features for players
        df_sium = feature_engineering_player(df_final)
        print(f"LENGTH OF ENGINEERED DATAFRAME: {len(df_sium.columns)}")
        
        # Drop features deemed useless to reduce noise
        df_sium = drop_useless_features(df_sium)
        print(f"LENGTH OF ELIMINATED DATAFRAME: {len(df_sium.columns)}")
        
        # Save the full featured player dataframe
        df_sium.to_pickle(os.path.join(final_folder, "FinalDF_player.pkl"))
        
        # SPLIT BY ROLE
        split_and_save_roles(df_sium, final_folder)

        # FINAL MERGE
        df_merged = merge_players_with_teams(df_sium, df_teams)
        df_merged.to_pickle(os.path.join(final_folder, "FinalDF_complete.pkl"))
        
        print(f"Completed {season}")

import pandas as pd
import os

# --- CONFIGURAZIONE ---
FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Il tuo percorso
# Nota: Ho escluso la stagione 2020-2021 dalla lista come da tue specifiche
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]

def count_statistics():
    print(f"--- ANALISI DIMENSIONI DATASET ---\n")
    
    # Inizializziamo i contatori globali
    global_unique_ids = set()
    total_raw_rows = 0
    
    # Dizionari per i raggruppamenti (Train/Val/Test)
    stats_by_split = {
        'Training': {'rows': 0, 'ids': set()},
        'Validation': {'rows': 0, 'ids': set()},
        'Test': {'rows': 0, 'ids': set()}
    }

    for season in SEASONS:
        file_path = os.path.join(FOLDER, f"season{season}", "FinalDF_player.pkl")
        
        if not os.path.exists(file_path):
            print(f"⚠️ File non trovato per {season}: {file_path}")
            continue
            
        # Carica il dataframe
        df = pd.read_pickle(file_path)
        
        # Filtra i portieri se non sono già stati rimossi (opzionale, ma sicuro)
        if 'Role' in df.columns:
            df = df[df['Role'] != 'P'] # Assumendo 'P' per portiere o 'GK'
        elif 'Pos' in df.columns:
            df = df[df['Pos'] != 'GK']

        # Calcoli per la stagione corrente
        n_rows = len(df)
        unique_players = df['Player_id'].unique()
        n_unique = len(unique_players)
        
        # Aggiorna i totali globali
        total_raw_rows += n_rows
        global_unique_ids.update(unique_players)
        
        print(f"📅 Stagione {season}: {n_rows} righe | {n_unique} giocatori unici")

        # --- ASSEGNAZIONE SPLIT (Logica basata sul tuo report) ---
        if season == "2023-2024":
            split = "Validation"
        elif season == "2024-2025":
            split = "Test"
        else:
            split = "Training"
            
        stats_by_split[split]['rows'] += n_rows
        stats_by_split[split]['ids'].update(unique_players)

    print("\n" + "="*40)
    print("   RISULTATI FINALI PER IL REPORT")
    print("="*40)
    
    print(f"\n🔢 TOTALE COMPLESSIVO (Intero Dataset):")
    print(f"   - Totale Osservazioni (Righe): {total_raw_rows}")
    print(f"   - Totale Giocatori Unici:      {len(global_unique_ids)}")
    
    print(f"\n📊 DETTAGLIO PER SPLIT:")
    for split_name, data in stats_by_split.items():
        # Calcolo approssimativo sequenze (righe - n_past * giocatori)
        # È una stima, per il numero esatto usa il codice delle sequenze
        print(f"   > {split_name}:")
        print(f"     - Righe Grezze:    {data['rows']}")
        print(f"     - Giocatori Unici: {len(data['ids'])}")


if __name__ == "__main__":
    count_statistics()
    # create_dfs()