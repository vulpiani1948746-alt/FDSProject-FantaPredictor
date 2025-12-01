from preclean_data import initial_data
from scraper_fbref import main_scraper
from scraper_fanta import extract_fanta
from merge_data import return_final_dataframe
from feature_engineering import merge_players_with_teams, feature_engineering_player
from feature_selection import drop_useless_features
from team_goals_predictor import TeamGoalsPredictor 
import pandas as pd
import os
import time


###### CHANGE THESE VALUES!!!
FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Put your folder that will contain all the data
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 

def create_dfs():
    predictor = TeamGoalsPredictor()

    for season in SEASONS:
        print(f"\n\n\nLet's do season {season}")
        print("-" * 50)
        print("\n\n")

        final_folder = os.path.join(FOLDER, f"season{season}")
        os.makedirs(final_folder, exist_ok=True)
        df_teams, df_stats = initial_data(season, final_folder)

        df_teams.to_pickle(os.path.join(final_folder, "teams.pkl"))
        df_stats.to_pickle(os.path.join(final_folder, "stats.pkl"))

        print(f"🤖 Calcolo previsioni Goal Squadra per {season}...")
        try:
            current_idx = SEASONS.index(season)
            seasons_so_far = SEASONS[:current_idx+1]
            
            df_aug = predictor.prepare_training_data(seasons=seasons_so_far, folder=FOLDER)
            
            if len(df_aug) > 50:
                used_feats = predictor.train(df_aug)
                df_with_preds = predictor.predict_and_save(df_aug, used_feats)
                
                df_preds_current = df_with_preds[df_with_preds['Season'] == season]
                
                df_preds_current = df_preds_current.drop_duplicates(subset=['Team_Name', 'Wk'])
                cols_to_merge = ['Team_Name', 'Wk', 'Pred_Team_Goals', 'Pred_Team_Conceded', 'Pred_Goal_Diff']
                
                if 'Pred_Team_Goals' in df_preds_current.columns:
                    df_teams = df_teams.merge(
                        df_preds_current[cols_to_merge], 
                        on=['Team_Name', 'Wk'], 
                        how='left'
                    )
                    print(f"✅ Previsioni integrate per {season}!")
            else:
                print(f"⚠️ Pochi dati per addestrare il modello (normale per {season}). Salto.")

        except Exception as e:
            print(f"⚠️ Errore nel calcolo previsioni (normale se mancano dati storici): {e}")

        for col in ['Pred_Team_Goals', 'Pred_Team_Conceded', 'Pred_Goal_Diff']:
            if col not in df_teams.columns:
                df_teams[col] = 0.0
        
        df_teams.to_pickle(os.path.join(final_folder, "teams.pkl"))

        print("DataFrame of Teams\n")
        print(df_teams.tail(20))
        print("-" * 50)
        print("\n")
        #time.sleep(10)


        print("DataFrame of players\n")
        #print(df_stats.head(20))
        print("-" * 50)
        print("\n")
        #time.sleep(10)


        print("Let's merge the two DataFrames...")
        df_final = return_final_dataframe(season, final_folder)
        #print(df_final[:20])
        print("-" * 50)

        df_sium = feature_engineering_player(df_final)
        df_sium = drop_useless_features(df_sium)
        df_sium.to_pickle(os.path.join(final_folder, "FinalDF_player.pkl"))
        print(df_sium.tail(20))

        df_merged = merge_players_with_teams(df_sium, df_teams)
        df_merged.to_pickle(os.path.join(final_folder, "FinalDF_complete.pkl"))
        print(df_merged.tail(20))


if __name__ == "__main__":
    create_dfs()
