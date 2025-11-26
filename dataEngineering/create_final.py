from preclean_data import initial_data
from scraper_fbref import main_scraper
from scraper_fanta import extract_fanta
from merge_data import return_final_dataframe
import pandas as pd
import os
import time


###### CHANGE THESE VALUES!!!
FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Put your folder that will contain all the data
SEASON = '2024-2025'  # Choose the season 

final_folder = os.path.join(FOLDER, f"season{SEASON}")
os.makedirs(final_folder, exist_ok=True)

### COMMENT OUT THIS PART FOR THE RE-RUNS -----------

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
main_scraper(SEASON, final_folder)
### UNTIL HERE -------------------------------

df_teams, df_stats = initial_data(SEASON, final_folder)

print("DataFrame of Teams\n")
print(df_teams.head(20))
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
print(df_final[:20])
print("-" * 50)