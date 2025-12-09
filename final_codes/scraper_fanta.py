import os
import requests
import time
import pandas as pd
import numpy as np

# Constants for the scraping logic
TOTAL_MATCHDAYS = 38
BASE_URL = "https://www.fantacalcio.it/api/v1/Excel/votes"

def fanta_scraper(output_folder, season):
    """
    Downloads Excel files for every matchday of a specific season from Fantacalcio.it.
    Also downloads the initial player values (quotazioni).
    """
    # Calculate the internal Season ID used by Fantacalcio.it API
    SEASON_ID = str(int(season[-2:])-6)
    
    # Hardcoded cookies required for authentication/session handling on the requests
    COOKIE = "cacheid=1717971175883.56679248262341275; _fbp=fb.1.1756830969795.359820317353317082; usprivacy=1---; _cc_id=323a40a10aebf55bc2fa29e12f330174; cto_bidid=qnDgEF9sRk1pRXpqeXAxcHJidEtnYU9iODFveVE2VG1oJTJCbUhuTE5aMm4wOUx6Q1JMZHpzS1hGTjFJJTJCbTFMR0QlMkIlMkZBRW84Q3ZVMktkTTZRU3RjNFBzVlpna0VwbzVLVzRJOFVjeDBKUFg1ekpOeXZTVnMwUTI4Tjk0ZWZ5UlNHdmI1dkxG; _ga_MHB3WGWHXC=GS2.1.s1757441051$o3$g1$t1757455346$j60$l0$h0; pbjs-unifiedid=%7B%22TDID_LOOKUP%22%3A%22FALSE%22%2C%22TDID_CREATED_AT%22%3A%222025-10-01T17%3A47%3A24%22%7D; pbjs-unifiedid_cst=3iwYLKIspg%3D%3D; cto_bundle=NfmetV9qdHlJNDNMNHhwcnZ2OUZxY1pxQ1FXOTZtTG8lMkJqZ2VqJTJGJTJGT3FEWHNmaTRoU2U5VkJGdENkSkkwMkFrMG93R3Z2dGp2cXNYR0liRTJjeGhFNnlIaktIV1RnUEFLSFdiYlNYSU9SRWltTFg5aUdNeW9QYUw2TGR3NzhJVU1KQld5TFM5YU1VTGg3T2o4MHlHT2Y4cyUyQkpmdyUzRCUzRA; cto_bundle=imjJfF83a1hibVlRJTJGN3NFaW4wUWY0Z3hDaGRtU1U1OGN1VmV3SmdMeDZWU3VSMzVBdDk0a2d6V3dSS2djYVAxQjFRaFVnUmdEQkJrQ1l0OTJhQWN5Uzk1TXFMV0p0akFpeWlkWUlZa2c4cWhaWCUyQkp1eGFjMnI3b2c3MVhUTTAzMHNSVDBFa0RUaHk1MFRmdTRZbzBBeEltJTJCNGclM0QlM0Q; _gid=GA1.2.1889255959.1764085629; euconsent-v2=CQbcW8AQbcW8AAKA9AITCGFgAAAAAEPgABCYAAAXgAAMBAAgiBIAQAAAAAIAACAAAAAgAAAAAgBQAAAiCAACBgAAIAEAIAAAAgAAgAAAgAAAAAQABAAAAACAQAgACBQABgAAAAAAMAAAGACQEAAABAMAxTAggECgACEiiCCAFAACCAEIAEAgGABACAAo4AggBAgUAAAIAAQAAACwEAgAICUgQABEQDAAAEAAAUQAgAIQsABBEAIBQRACcRgKIBg0IAklKgyAAgDAyTIhE-EQMQgBCgBSA2KEYA6eAKEg4gALgAoAC4AHAAPAAggBeAGoAPAAhgBMACqAG8APwAhIBDAESAI4ATQArQBgADDgGUAZYA2QBzwDuAO-AewB8QD7AP0AgABFICLgIxARoBHICRAJNAT8BQYCoAKuAXMAvQBigDRAG0ANwAcQA9oCHYEegSKAmkBOwChwFHgKRAU2AtgBcAC5AF2ALvAXmAw2BkYGSAMnAZcAzMBnMDVwNZAbGA2gBt4DcwG6gOTAcuA8cB8YD_gICAQTAgwBCGCFoIXwQ9BD8CPoEioJMAkyBLMCW8EvgTAAmWBM4CaoE2AJuQTmBOkCdwE8IKDgoQBQsCioFJQKXAUyAp5BT8FQAKjAVKAqcBV0CtgFbwK-AWFAscBZUC00FrAWtAtyBcUdCEAAXABQAFQAOAAggBcAGoAPAAmABTACrAFwAXQAxABvAD9AIYAiQBHACaAFGAK0AYAAwwBlADRAGyAOeAdwB3wD2APiAfYB-wEUARiAjgCTQE_AUGAqICrgFiALnAXkBegDFAG0ANxAcQBxgD2gH2AQgAh0BE4CL4EegSKAmQBNICdgFDwKPApABSYCmwFWALFAWwAt0BcAC5IF2AXaAu8BeYC-gGGgMegZGBkkDJwMqgZYBlwDMwGcgNNAarA1cDWAG0ANvAbqA4sByYDlwHjgPjAfWA-4B_YD_gIAgQEAgwBC0CHoEdgI-gSEAkUBJkCVcEswS0Al1BL4EwAJlgTOAmqBNgCbgE4YJzAnTBO0E7gJ4ATzAoMBQgChYFEgKKgUlApcBTwCn4FRgKlAVOgq4CroFbAK3gWDAsmBZWC1gLWgW5At2BcUC44FyyED8ABYAFAAXAA1ACqAFwAMQAbwA_ADAAHPAO4A7wCKAEcAJSAUGAqICrgFzAMUAbQA4wCHQETgI9ATSApMBTYCrAFigLRAXAAuQBdgDIwGTgM5AaqA8cB8YD-wICAQYAhaBD0CQgEigJdATOAmwBOYCdwE8AJ5gUGAoqBSUClwFPwKlAVfAsEBa0C3JKCgAAsACgAHAAeABDACYAFUALgAYoBDAESAI4AUYArQBgADZAHeAPwAjkBUQFXALmAYoA4gCEAEOgIvgR6BIoCjwFJwKaApsBYoC2AFyALsAXnAyMDJAGTgMsAZyA1gBt4D4wIAgQEAgeBBgCEIEPQI-gSKAkqBKsCVoEugJfATOAmqBNgCbgE4QJzQTtBO4CeAE8wKKgUlApcBTwCowFSgKnAVsArfBYIFgwLFgWVAs9BawFrQLcgXIAuWUg3gALgAoACoAHAAQQAyADQAHgAQwAmABVADEAH6AQwBEgCOAFGAK0AYAAygBogDZAHOAO-AfYB-gEWAIxARwBJQCgwFRAVcAuYBeQDFAG0ANxAcQBxgD2gH2AQ6AicBF8CPQJFATSAnYBQ4CkAFJwKaApsBVgCxQFsALgAXJAuwC7QF5gL6AYbAyMDJAGTwMsAy4BnMDWANZAbeA3UByYDxQHjgPjAf2A_4CAgEEwIMAQhAhaBDMCHIEdgI-gSKgkwCTIEqwJZgS6gl8CYAEywJnATVAmwBOaCdoJ3ATwAnmBQsCioFJQKXAU8Ap-BUYCpQFTgK2AVvAsEBYUCxYFkwLKgWegtYC1oFuQLigXIA.YAAAAAAAAAAA; fantacalcio.it=uZczj00LpQ26Kb7ntVodgya4Co99jgjt8iHtFGBIRKB71Y5H4byIR7hTMXrJUmKmCSwhZBA%2BCOoToYZEUGu5%2F0nf0UaX532BHxg7PZsRn%2FB4gVnPWMgmFUuw%2F0V30eQLr15hIXXUqQSU6JlVvHGMhmK%2BmBph5g8Ag3%2BNmXIZlblgr6m%2FSXVHOdyQ%2BiTup%2BcMFIs2Y6mzlp4aFg3uf7m%2FH9qflHsCHSCQocOQIQ%2FlG6IdxiCEtvz5kBZI3%2BNzIG6pcTOW%2BSsEd53nEKPAYCM9mo12StL23dtIfbXLxeFZoA068%2Bv7HuFdTfWx4%2BWupmkOG%2ByCouFBVKqtDlhHjrv1cSOY8WHWWS1UwaO30vR43lVEgGzqhpEPuXGXUugVK%2FP67uDO1HStAd815A5PsrYJHDbjIGl9sxEW4C1pFT7j0qf9yO8GVD3%2BNbhTMXrJUmKmCSwhZBA%2BCOoToYZEUGu5%2F0nf0UaX532B9Wc5sTZ%2Fjow%3D; client.fantacalcio.it=francescofinazzi02%40gmail.com; comp_fantacalcio-storico_FCLeague2025=0=438785; euconsent-v2=CQbcW8AQbcW8AFgAGAITCGFsAP_gAAAAABCYLxtR_G__bXlr-bb36ftkeYxf9_hr7sQxBgbJk24FzLvW7JwX32E7NAzatqYKmRIAu3TBIQNlHJDURVCgKIgFrzDMaEyUoTtKJ6BkiFMRY2NYCFxvm4tjWQCY5vr99ld9mR-N7dr82dzyy6hnv3a9_-S1WJCdIYetDfv8ZBKT-9IE9_x8v4v4_N7pE2-eS1n_tGvp4j9-YvP_dBmxt-bSffzPn__rl_e7X__d_n37v94XX77_____f_-7___2b4AAACgkAGAAILxhoAMAAQXjEQAYAAgvGKgAwABBeMZABgACC8Y6ADAAEF4yEAGAAILxkoAMAAQXjKQAYAAgvGWgAwABBeMA.ILxtR_G__bXlv-bb36ftkeYxf9_hr7sQxBgbJs24FzLvW7JwX32E7NEzatqYKmRIAu3TBIQNtHJjURVChKIgVrzDsaEyUoTtKJ-BkiHMRY2NYCFxvm4tjWQCZ5vr_91d9mT-N7dr-2dzyy7hnv3a9_-S1WJidKYetHfv8ZBKT-_IU9_x-_4v4_N7pE2-eS1v_tWvt439-4vP_dpuxt-byffz____73_e7X__d_______Xf_7_____________f4A.f_wAAAAAAAAA; pubtech-cmp-pcstring=0-111; __gads=ID=c326da87fd2f9951:T=1759340840:RT=1764097201:S=ALNI_MavNSS1u5o9fvzxFbhp8RkrtUEGrw; __gpi=UID=000012932d0324bb:T=1759340840:RT=1764097201:S=ALNI_MZoN2iGPA1ghWVcatfUrmlLn-F5Mg; __eoi=ID=4cfbc3f08644bfc5:T=1756830969:RT=1764097201:S=AA-AfjYCH6I6zcxjjVdhPAUImAYW; cto_bundle=hfPc0l8zRiUyRmpaYmJuVGloTmx6blFiMWJTOWNNWkFPTzFES2dmY2tIb2pYc0VlUFhSUGhQUjVvRXVWbnRJYk5hSHlyYjJxVEYxUXE2OGRxbEMyUnlMTVVlVUNOUGNBTjNVRTlHTGJMMFNoa1FZcUI2MyUyQiUyRk5TSnNMbHNiNUVLVlNMTHM5T3olMkJ0NVVsSCUyQjAwTHdvVXFZd0VVVjdBJTNEJTNE; tncid=a732ead6-df87-46f0-8813-50f3295d7218; _ga_DVR9V968VW=GS2.1.s1764097199$o3$g0$t1764097202$j57$l0$h0; _ga_KVFYCX9ZTZ=GS2.1.s1764182687$o10$g1$t1764182814$j15$l0$h0; _ga=GA1.1.14977322.1756830969"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Standard headers with User-Agent to mimic a browser
    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.fantacalcio.it/voti-fantacalcio-serie-a",
            "Cookie": COOKIE
        }
    
    print(f"Inizio download per la stagione ID {SEASON_ID}...\n")

    # DOWNLOAD MATCHDAY VOTES
    for giornata in range(1, TOTAL_MATCHDAYS + 1):
        url = f"{BASE_URL}/{SEASON_ID}/{giornata}"
        filename = os.path.join(output_folder, f"Voti_Giornata_{giornata}.xlsx")

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                # Basic check to avoid saving empty/broken files
                if len(response.content) > 1000:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print(f"Week {giornata}: Correctly Downloaded.")
                else:
                    print(f"Week {giornata}: Empty File.")
            elif response.status_code == 401:
                print(f"❌ Week {giornata}: Error 401. Wrong or expired cookies.")
                break 
            elif response.status_code == 404:
                print(f"❌ Week {giornata}: Not available.")
            else:
                print(f"❌ Week {giornata}: Error HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️  Error: {e}")
        
        time.sleep(1)
    
    # DOWNLOAD PLAYER VALUES
    url_quotazioni = f"https://www.fantacalcio.it/api/v1/Excel/prices/{SEASON_ID}/1"
    filename_quotazioni = os.path.join(output_folder, "Fantavalori.xlsx")

    try:
        response = requests.get(url_quotazioni, headers=headers)
        
        if response.status_code == 200:
            if len(response.content) > 1000:
                with open(filename_quotazioni, "wb") as f:
                    f.write(response.content)
                print(f"Values correctly downloaded to: {filename_quotazioni}")
            else:
                print(f"Values file too small or inexistent.")
        elif response.status_code == 404:
            print(f"❌ Values file not found.")
        else:
            print(f"❌ Error download of values file: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️  Error in downloading the values file: {e}")

    print("\nProcess terminated.")


def clean_fantacalcio_data(df, vote_column_name):
    """
    Cleans raw dataframe from Fantacalcio.it Excel sheets.
    Handles missing Team Names, renames columns, extracts roles, 
    and calculates the 'Fantavoto' (Vote + Bonuses - Maluses).
    """
    
    # FILL TEAM NAMES: The Excel usually puts the Team Name only in the first row of the team's block.
    mask_team = df.iloc[:, 1].isna()
    df['Squadra'] = df.iloc[:, 0].where(mask_team).ffill()
    
    # Rename generic columns to meaningful names based on file structure
    col_map = {
        0: 'Cod', 1: 'Ruolo', 2: 'Nome', 3: 'Voto',
        4: 'Gf', 5: 'Gs', 6: 'Rp', 7: 'Rs', 8: 'Rf', 9: 'Au', 
        10: 'Amm', 11: 'Esp', 12: 'Ass'
    }
    df = df.rename(columns=col_map)

    # Filter out header rows/junk by checking if 'Cod' is numeric
    df = df[pd.to_numeric(df['Cod'], errors='coerce').notnull()]
    df['Cod'] = df['Cod'].astype(int)

    df = df.rename(columns={'Voto': vote_column_name})

    # Filter for outfield players (Defenders, Midfielders, Attackers)
    df['Ruolo'] = df['Ruolo'].astype(str).str.strip().str.upper()
    df = df[df['Ruolo'].isin(['D', 'C', 'A'])]
    
    df['Squadra'] = df['Squadra'].astype(str).str.strip()
    
    # Convert bonus columns to numeric, filling NaNs with 0
    bonus_cols = ['Gf', 'Gs', 'Rp', 'Rs', 'Rf', 'Au', 'Amm', 'Esp', 'Ass']
    for col in bonus_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Combined Penalties Missed (Saved + Off target)
    df['Rigori_Sbagliati'] = df['Rs'] + df['Rp']
    
    # Clean vote column (remove asterisks used for subbed/special cases)
    voto_numeric = df[vote_column_name].astype(str).str.replace('*', '', regex=False)
    voto_numeric = pd.to_numeric(voto_numeric, errors='coerce')
    
    # --- FANTAVOTO CALCULATION ---
    df['Fantavoto'] = voto_numeric + \
                      (df['Gf'] * 3) + \
                      (df['Rf'] * 3) + \
                      (df['Rp'] * 3) + \
                      (df['Ass'] * 1) - \
                      (df['Gs'] * 1) - \
                      (df['Au'] * 2) - \
                      (df['Rs'] * 3) - \
                      (df['Amm'] * 0.5) - \
                      (df['Esp'] * 1)

    return df[['Cod', 'Ruolo', 'Nome', 'Squadra', vote_column_name, 'Au', 'Rigori_Sbagliati', 'Fantavoto']]


def single_file(file_path):
    """
    Processes a single Excel file (one matchday).
    Reads 3 sheets (Fantacalcio, Statistico, Italia), cleans them, 
    and merges them into one row per player.
    """
    try:
        # Read raw data from 3 different editorial sources (sheets)
        df_fanta_raw = pd.read_excel(file_path, sheet_name="Fantacalcio", skiprows=4, header=None)
        df_stat_raw = pd.read_excel(file_path, sheet_name="Statistico", skiprows=4, header=None)
        df_ita_raw = pd.read_excel(file_path, sheet_name="Italia", skiprows=4, header=None)

        # Clean each source independently
        df_fanta = clean_fantacalcio_data(df_fanta_raw, 'Voto_Fantacalcio')
        df_stat = clean_fantacalcio_data(df_stat_raw, 'Voto_Statistico')
        df_ita = clean_fantacalcio_data(df_ita_raw, 'Voto_Italia')
        
        if df_fanta is None or df_fanta.empty: return None

        # Merge secondary sources into the main Fantacalcio dataframe
        df_final = df_fanta.merge(df_stat[['Cod', 'Voto_Statistico']], on='Cod', how='left')
        df_final = df_final.merge(df_ita[['Cod', 'Voto_Italia']], on='Cod', how='left')

        # 'sv' (senza voto) flag calculation based on asterisk
        df_final['sv'] = np.where(df_final['Voto_Fantacalcio'].astype(str).str.endswith('*'), 1, 0)

        # Final cleanup of vote columns
        cols_voti = ["Fantacalcio", "Statistico", "Italia"]
        for col in cols_voti:
            column = f"Voto_{col}"
            df_final[column] = df_final[column].astype(str).str.replace('*', '', regex=False)
            df_final[column] = pd.to_numeric(df_final[column], errors='coerce')

        return df_final
    
    except Exception as e:
        print(f"Error file {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()


def integra_quotazioni_per_codice(df_fanta, path_quotazioni):
    """
    Loads the player values (Quotazioni) from Excel and merges them 
    into the main dataframe using 'Cod' as key.
    Handles both active players ('Tutti') and transferred players ('Ceduti').
    """
    try:
        df_tutti = pd.read_excel(path_quotazioni, sheet_name='Tutti', header=1)
        
        # Try loading 'Ceduti' sheet if players were transferred mid-season
        try:
            df_ceduti = pd.read_excel(path_quotazioni, sheet_name='Ceduti', header=1)
            df_valori = pd.concat([df_tutti, df_ceduti], ignore_index=True)
        except:
            print("⚠️ Foglio 'Ceduti' non trovato, uso solo 'Tutti'.")
            df_valori = df_tutti

        df_valori.columns = df_valori.columns.str.strip()
        
        # Identify the Initial Quote column (usually contains "Qt" and "I")
        col_qt_i = None
        for col in df_valori.columns:
            if "Qt" in col and "I" in col: 
                col_qt_i = col
                break
        
        if not col_qt_i:
            print("❌ Error: Column 'Qt.I.' not found.")
            return df_fanta

        # Select only ID and Quote
        df_valori_clean = df_valori[['Id', col_qt_i]].copy()
        df_valori_clean = df_valori_clean.rename(columns={'Id': 'Cod', col_qt_i: 'Qt.I'})
        
        # Clean IDs
        df_valori_clean['Cod'] = pd.to_numeric(df_valori_clean['Cod'], errors='coerce')
        df_valori_clean = df_valori_clean.dropna(subset=['Cod'])
        df_valori_clean['Cod'] = df_valori_clean['Cod'].astype(int)
        
        # Deduplicate to avoid exploding the merge
        df_valori_clean = df_valori_clean.drop_duplicates(subset=['Cod'])

        # Merge Quotes into the main DF
        df_merged = df_fanta.merge(df_valori_clean, on='Cod', how='left')
        
        # Default missing quotes to 1
        df_merged['Qt.I'] = df_merged['Qt.I'].fillna(1).astype(int)
        
        print(f"✅ Values successuflly merged! Total rows: {len(df_merged)}")
        return df_merged

    except Exception as e:
        print(f"❌ Error in the values integration: {e}")
        return df_fanta


def extract_fanta(folder, season):
    """
    Orchestrator function:
    1. Downloads data via fanta_scraper.
    2. Iterates over all downloaded files to clean and merge data.
    3. Integrates player values.
    4. Saves the final result.
    """
    base_folder = os.path.join(folder, f"voti_fanta_{season}") 
    os.makedirs(base_folder, exist_ok=True)
    
    # RUN SCRAPER
    fanta_scraper(base_folder, season)
    
    all_dfs = []
    print(f"Extraction of data from: {base_folder}")

    # PROCESS FILES
    for i in range(1, TOTAL_MATCHDAYS+1):
        file_path = os.path.join(base_folder, f'Voti_Giornata_{i}.xlsx')
        
        if os.path.exists(file_path):
            df_giornata = single_file(file_path)
            
            if not df_giornata.empty:
                df_giornata["id_giornata"] = i
                all_dfs.append(df_giornata)
    
    if all_dfs:
        total_df = pd.concat(all_dfs, ignore_index=True)

        # INTEGRATE VALUES
        file_quotazioni = os.path.join(base_folder, "Fantavalori.xlsx")
        final_df = integra_quotazioni_per_codice(total_df, file_quotazioni)
        
        # SAVE FINAL
        final_df.to_excel(os.path.join(folder, "Fanta_Total.xlsx"), index=False)
        return final_df
    else:
        return pd.DataFrame()