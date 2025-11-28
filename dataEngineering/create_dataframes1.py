from preclean_data import initial_data
from scraper_fbref import main_scraper
from scraper_fanta import extract_fanta
from merge_data import return_final_dataframe
from feature_engineering import merge_players_with_teams, feature_engineering_player
from feature_selection import drop_useless_features
import pandas as pd
import os
import time


###### CHANGE THESE VALUES!!!
FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Put your folder that will contain all the data
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 

def create_dfs():
    for season in SEASONS:
        print(f"\n\n\nLet's do season {season}")
        print("-" * 50)
        print("\n\n")

        final_folder = os.path.join(FOLDER, f"season{season}")
        os.makedirs(final_folder, exist_ok=True)
        df_teams, df_stats = initial_data(season, final_folder)

        df_teams.to_pickle(os.path.join(final_folder, "teams.pkl"))
        df_stats.to_pickle(os.path.join(final_folder, "stats.pkl"))

        print("DataFrame of Teams\n")
        #print(df_teams.tail(20))
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

create_dfs()
