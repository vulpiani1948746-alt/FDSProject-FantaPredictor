from preclean_data import initial_data
from scraper_fbref import main_scraper
from scraper_fanta import extract_fanta
from merge_data import return_final_dataframe
from feature_engineering import merge_players_with_teams, feature_engineering_player
from create_time_series import extract_player
from feature_selection import correlation_matrix
import first_lstm
import pandas as pd
import os
import time
pd.set_option('display.max_rows', None)


###### CHANGE THESE VALUES!!!
FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Put your folder that will contain all the data
SEASON = '2024-2025'  # Choose the season 

final_folder = os.path.join(FOLDER, f"season{SEASON}")
os.makedirs(final_folder, exist_ok=True)

### COMMENT OUT THIS PART FOR THE RE-RUNS -----------
"""
print("Scrape Data, create files and DatFrame for Fantasy Football...")
print("Change the Cookie before doing this!!!")
df_fanta = extract_fanta(final_folder, SEASON)
df_fanta.to_pickle(f"{final_folder}/Fantacalcio.pkl")

print("DataFrame of Fantasy Football\n")
print(df_fanta.head(20))
print("-" * 50)
print("\n")
#time.sleep(10)

print("Scrape Data, create file and DataFrame for stats of players...")
main_scraper(SEASON, final_folder)"""
### UNTIL HERE -------------------------------

df_teams, df_stats = initial_data(SEASON, final_folder)

"""
print("DataFrame of Teams\n")
print(df_teams.tail(20))
print("-" * 50)
print("\n")
#time.sleep(10)


print("DataFrame of players\n")
print(df_stats.head(20))
print("-" * 50)
print("\n")
#time.sleep(10)


print("Let's merge the two DataFrames...")
df_final = return_final_dataframe(SEASON, final_folder)
df_final.to_pickle(os.path.join(final_folder, "FinalDF.pkl"))
print(df_final[:20])
print("-" * 50)
#df_final = pd.read_pickle(os.path.join(final_folder, "FinalDF.pkl"))

df_merged = merge_players_with_teams(df_final, df_teams)
df_sium = feature_engineering_player(df_merged)
print(df_sium.tail(20))"""

df_teams = pd.read_pickle(os.path.join(final_folder, "teams.pkl"))
df_players = pd.read_pickle(os.path.join(final_folder, "FinalDF_player.pkl"))

df = first_lstm.prepare_data_dual_input(df_players, df_teams)[0]
print(df.tail(20))