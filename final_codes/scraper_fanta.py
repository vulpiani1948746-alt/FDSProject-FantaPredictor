import os
import requests
import time
import pandas as pd
import numpy as np

TOTAL_MATCHDAYS = 38
BASE_URL = "https://www.fantacalcio.it/api/v1/Excel/votes"

def fanta_scraper(output_folder, season):
    SEASON_ID = str(int(season[-2:])-6)
    COOKIE = "cacheid=1717971175883.56679248262341275; _fbp=fb.1.1756830969795.359820317353317082; usprivacy=1---; _cc_id=323a40a10aebf55bc2fa29e12f330174; cto_bidid=qnDgEF9sRk1pRXpqeXAxcHJidEtnYU9iODFveVE2VG1oJTJCbUhuTE5aMm4wOUx6Q1JMZHpzS1hGTjFJJTJCbTFMR0QlMkIlMkZBRW84Q3ZVMktkTTZRU3RjNFBzVlpna0VwbzVLVzRJOFVjeDBKUFg1ekpOeXZTVnMwUTI4Tjk0ZWZ5UlNHdmI1dkxG; _ga_MHB3WGWHXC=GS2.1.s1757441051$o3$g1$t1757455346$j60$l0$h0; pbjs-unifiedid=%7B%22TDID_LOOKUP%22%3A%22FALSE%22%2C%22TDID_CREATED_AT%22%3A%222025-10-01T17%3A47%3A24%22%7D; pbjs-unifiedid_cst=3iwYLKIspg%3D%3D; cto_bundle=NfmetV9qdHlJNDNMNHhwcnZ2OUZxY1pxQ1FXOTZtTG8lMkJqZ2VqJTJGJTJGT3FEWHNmaTRoU2U5VkJGdENkSkkwMkFrMG93R3Z2dGp2cXNYR0liRTJjeGhFNnlIaktIV1RnUEFLSFdiYlNYSU9SRWltTFg5aUdNeW9QYUw2TGR3NzhJVU1KQld5TFM5YU1VTGg3T2o4MHlHT2Y4cyUyQkpmdyUzRCUzRA; cto_bundle=imjJfF83a1hibVlRJTJGN3NFaW4wUWY0Z3hDaGRtU1U1OGN1VmV3SmdMeDZWU3VSMzVBdDk0a2d6V3dSS2djYVAxQjFRaFVnUmdEQkJrQ1l0OTJhQWN5Uzk1TXFMV0p0akFpeWlkWUlZa2c4cWhaWCUyQkp1eGFjMnI3b2c3MVhUTTAzMHNSVDBFa0RUaHk1MFRmdTRZbzBBeEltJTJCNGclM0QlM0Q; _gid=GA1.2.1889255959.1764085629; euconsent-v2=CQbcW8AQbcW8AAKA9AITCGFgAAAAAEPgABCYAAAXgAAMBAAgiBIAQAAAAAIAACAAAAAgAAAAAgBQAAAiCAACBgAAIAEAIAAAAgAAgAAAgAAAAAQABAAAAACAQAgACBQABgAAAAAAMAAAGACQEAAABAMAxTAggECgACEiiCCAFAACCAEIAEAgGABACAAo4AggBAgUAAAIAAQAAACwEAgAICUgQABEQDAAAEAAAUQAgAIQsABBEAIBQRACcRgKIBg0IAklKgyAAgDAyTIhE-EQMQgBCgBSA2KEYA6eAKEg4gALgAoAC4AHAAPAAggBeAGoAPAAhgBMACqAG8APwAhIBDAESAI4ATQArQBgADDgGUAZYA2QBzwDuAO-AewB8QD7AP0AgABFICLgIxARoBHICRAJNAT8BQYCoAKuAXMAvQBigDRAG0ANwAcQA9oCHYEegSKAmkBOwChwFHgKRAU2AtgBcAC5AF2ALvAXmAw2BkYGSAMnAZcAzMBnMDVwNZAbGA2gBt4DcwG6gOTAcuA8cB8YD_gICAQTAgwBCGCFoIXwQ9BD8CPoEioJMAkyBLMCW8EvgTAAmWBM4CaoE2AJuQTmBOkCdwE8IKDgoQBQsCioFJQKXAUyAp5BT8FQAKjAVKAqcBV0CtgFbwK-AWFAscBZUC00FrAWtAtyBcUdCEAAXABQAFQAOAAggBcAGoAPAAmABTACrAFwAXQAxABvAD9AIYAiQBHACaAFGAK0AYAAwwBlADRAGyAOeAdwB3wD2APiAfYB-wEUARiAjgCTQE_AUGAqICrgFiALnAXkBegDFAG0ANxAcQBxgD2gH2AQgAh0BE4CL4EegSKAmQBNICdgFDwKPApABSYCmwFWALFAWwAt0BcAC5IF2AXaAu8BeYC-gGGgMegZGBkkDJwMqgZYBlwDMwGcgNNAarA1cDWAG0ANvAbqA4sByYDlwHjgPjAfWA-4B_YD_gIAgQEAgwBC0CHoEdgI-gSEAkUBJkCVcEswS0Al1BL4EwAJlgTOAmqBNgCbgE4YJzAnTBO0E7gJ4ATzAoMBQgChYFEgKKgUlApcBTwCn4FRgKlAVOgq4CroFbAK3gWDAsmBZWC1gLWgW5At2BcUC44FyyED8ABYAFAAXAA1ACqAFwAMQAbwA_ADAAHPAO4A7wCKAEcAJSAUGAqICrgFzAMUAbQA4wCHQETgI9ATSApMBTYCrAFigLRAXAAuQBdgDIwGTgM5AaqA8cB8YD-wICAQYAhaBD0CQgEigJdATOAmwBOYCdwE8AJ5gUGAoqBSUClwFPwKlAVfAsEBa0C3JKCgAAsACgAHAAeABDACYAFUALgAYoBDAESAI4AUYArQBgADZAHeAPwAjkBUQFXALmAYoA4gCEAEOgIvgR6BIoCjwFJwKaApsBYoC2AFyALsAXnAyMDJAGTgMsAZyA1gBt4D4wIAgQEAgeBBgCEIEPQI-gSKAkqBKsCVoEugJfATOAmqBNgCbgE4QJzQTtBO4CeAE8wKKgUlApcBTwCowFSgKnAVsArfBYIFgwLFgWVAs9BawFrQLcgXIAuWUg3gALgAoACoAHAAQQAyADQAHgAQwAmABVADEAH6AQwBEgCOAFGAK0AYAAygBogDZAHOAO-AfYB-gEWAIxARwBJQCgwFRAVcAuYBeQDFAG0ANxAcQBxgD2gH2AQ6AicBF8CPQJFATSAnYBQ4CkAFJwKaApsBVgCxQFsALgAXJAuwC7QF5gL6AYbAyMDJAGTwMsAy4BnMDWANZAbeA3UByYDxQHjgPjAf2A_4CAgEEwIMAQhAhaBDMCHIEdgI-gSKgkwCTIEqwJZgS6gl8CYAEywJnATVAmwBOaCdoJ3ATwAnmBQsCioFJQKXAU8Ap-BUYCpQFTgK2AVvAsEBYUCxYFkwLKgWegtYC1oFuQLigXIA.YAAAAAAAAAAA; addtl_consent=1~; IABGPP_HDR_GppString=DBABMA~CQbehrtQbehrtAKA9AITCGFgAAAAAEPgABCYAAAXgAAMBAAgiBIAQAAAAAIAACAAAAAgAAAAAgBQAAAiCAACBgAAIAEAIAAAAgAAgAAAgAAAAAQABAAAAACAQAgACBQABgAAAAAAMAAAGACQEAAABAMAxTAggECgACEiiCCAFAACCAEIAEAgGABACAAo4AggBAgUAAAIAAQAAACwEAgAICUgQABEQDAAAEAAAUQAgAIQsABBEAIBQRACcRgKIBg0IAklKgyAAgDAyTIhE-EQMQgBCgBSA2KEYA6eAKEg4gALgAoAC4AHAAPAAggBeAGoAPAAhgBMACqAG8APwAhIBDAESAI4ATQArQBgADDgGUAZYA2QBzwDuAO-AewB8QD7AP0AgABFICLgIxARoBHICRAJNAT8BQYCoAKuAXMAvQBigDRAG0ANwAcQA9oCHYEegSKAmkBOwChwFHgKRAU2AtgBcAC5AF2ALvAXmAw2BkYGSAMnAZcAzMBnMDVwNZAbGA2gBt4DcwG6gOTAcuA8cB8YD_gICAQTAgwBCGCFoIXwQ9BD8CPoEioJMAkyBLMCW8EvgTAAmWBM4CaoE2AJuQTmBOkCdwE8IKDgoQBQsCioFJQKXAUyAp5BT8FQAKjAVKAqcBV0CtgFbwK-AWFAscBZUC00FrAWtAtyBcUdCEAAXABQAFQAOAAggBcAGoAPAAmABTACrAFwAXQAxABvAD9AIYAiQBHACaAFGAK0AYAAwwBlADRAGyAOeAdwB3wD2APiAfYB-wEUARiAjgCTQE_AUGAqICrgFiALnAXkBegDFAG0ANxAcQBxgD2gH2AQgAh0BE4CL4EegSKAmQBNICdgFDwKPApABSYCmwFWALFAWwAt0BcAC5IF2AXaAu8BeYC-gGGgMegZGBkkDJwMqgZYBlwDMwGcgNNAarA1cDWAG0ANvAbqA4sByYDlwHjgPjAfWA-4B_YD_gIAgQEAgwBC0CHoEdgI-gSEAkUBJkCVcEswS0Al1BL4EwAJlgTOAmqBNgCbgE4YJzAnTBO0E7gJ4ATzAoMBQgChYFEgKKgUlApcBTwCn4FRgKlAVOgq4CroFbAK3gWDAsmBZWC1gLWgW5At2BcUC44FyyED8ABYAFAAXAA1ACqAFwAMQAbwA_ADAAHPAO4A7wCKAEcAJSAUGAqICrgFzAMUAbQA4wCHQETgI9ATSApMBTYCrAFigLRAXAAuQBdgDIwGTgM5AaqA8cB8YD-wICAQYAhaBD0CQgEigJdATOAmwBOYCdwE8AJ5gUGAoqBSUClwFPwKlAVfAsEBa0C3JKCgAAsACgAHAAeABDACYAFUALgAYoBDAESAI4AUYArQBgADZAHeAPwAjkBUQFXALmAYoA4gCEAEOgIvgR6BIoCjwFJwKaApsBYoC2AFyALsAXnAyMDJAGTgMsAZyA1gBt4D4wIAgQEAgeBBgCEIEPQI-gSKAkqBKsCVoEugJfATOAmqBNgCbgE4QJzQTtBO4CeAE8wKKgUlApcBTwCowFSgKnAVsArfBYIFgwLFgWVAs9BawFrQLcgXIAuWUg3gALgAoACoAHAAQQAyADQAHgAQwAmABVADEAH6AQwBEgCOAFGAK0AYAAygBogDZAHOAO-AfYB-gEWAIxARwBJQCgwFRAVcAuYBeQDFAG0ANxAcQBxgD2gH2AQ6AicBF8CPQJFATSAnYBQ4CkAFJwKaApsBVgCxQFsALgAXJAuwC7QF5gL6AYbAyMDJAGTwMsAy4BnMDWANZAbeA3UByYDxQHjgPjAf2A_4CAgEEwIMAQhAhaBDMCHIEdgI-gSKgkwCTIEqwJZgS6gl8CYAEywJnATVAmwBOaCdoJ3ATwAnmBQsCioFJQKXAU8Ap-BUYCpQFTgK2AVvAsEBYUCxYFkwLKgWegtYC1oFuQLigXIA.YAAAAAAAAAAA; fantacalcio.it=uZczj00LpQ26Kb7ntVodgya4Co99jgjt8iHtFGBIRKB71Y5H4byIR7hTMXrJUmKmCSwhZBA%2BCOoToYZEUGu5%2F0nf0UaX532BHxg7PZsRn%2FB4gVnPWMgmFUuw%2F0V30eQLr15hIXXUqQSU6JlVvHGMhmK%2BmBph5g8Ag3%2BNmXIZlblgr6m%2FSXVHOdyQ%2BiTup%2BcMFIs2Y6mzlp4aFg3uf7m%2FH9qflHsCHSCQocOQIQ%2FlG6IdxiCEtvz5kBZI3%2BNzIG6pcTOW%2BSsEd53nEKPAYCM9mo12StL23dtIfbXLxeFZoA068%2Bv7HuFdTfWx4%2BWupmkOG%2ByCouFBVKqtDlhHjrv1cSOY8WHWWS1UwaO30vR43lVEgGzqhpEPuXGXUugVK%2FP67uDO1HStAd815A5PsrYJHDbjIGl9sxEW4C1pFT7j0qf9yO8GVD3%2BNbhTMXrJUmKmCSwhZBA%2BCOoToYZEUGu5%2F0nf0UaX532B9Wc5sTZ%2Fjow%3D; client.fantacalcio.it=francescofinazzi02%40gmail.com; comp_fantacalcio-storico_FCLeague2025=0=438785; euconsent-v2=CQbcW8AQbcW8AFgAGAITCGFsAP_gAAAAABCYLxtR_G__bXlr-bb36ftkeYxf9_hr7sQxBgbJk24FzLvW7JwX32E7NAzatqYKmRIAu3TBIQNlHJDURVCgKIgFrzDMaEyUoTtKJ6BkiFMRY2NYCFxvm4tjWQCY5vr99ld9mR-N7dr82dzyy6hnv3a9_-S1WJCdIYetDfv8ZBKT-9IE9_x8v4v4_N7pE2-eS1n_tGvp4j9-YvP_dBmxt-bSffzPn__rl_e7X__d_n37v94XX77_____f_-7___2b4AAACgkAGAAILxhoAMAAQXjEQAYAAgvGKgAwABBeMZABgACC8Y6ADAAEF4yEAGAAILxkoAMAAQXjKQAYAAgvGWgAwABBeMA.ILxtR_G__bXlv-bb36ftkeYxf9_hr7sQxBgbJs24FzLvW7JwX32E7NEzatqYKmRIAu3TBIQNtHJjURVChKIgVrzDsaEyUoTtKJ-BkiHMRY2NYCFxvm4tjWQCZ5vr_91d9mT-N7dr-2dzyy7hnv3a9_-S1WJidKYetHfv8ZBKT-_IU9_x-_4v4_N7pE2-eS1v_tWvt439-4vP_dpuxt-byffz____73_e7X__d_______Xf_7_____________f4A.f_wAAAAAAAAA; pubtech-cmp-pcstring=0-111; __gads=ID=c326da87fd2f9951:T=1759340840:RT=1764097201:S=ALNI_MavNSS1u5o9fvzxFbhp8RkrtUEGrw; __gpi=UID=000012932d0324bb:T=1759340840:RT=1764097201:S=ALNI_MZoN2iGPA1ghWVcatfUrmlLn-F5Mg; __eoi=ID=4cfbc3f08644bfc5:T=1756830969:RT=1764097201:S=AA-AfjYCH6I6zcxjjVdhPAUImAYW; cto_bundle=hfPc0l8zRiUyRmpaYmJuVGloTmx6blFiMWJTOWNNWkFPTzFES2dmY2tIb2pYc0VlUFhSUGhQUjVvRXVWbnRJYk5hSHlyYjJxVEYxUXE2OGRxbEMyUnlMTVVlVUNOUGNBTjNVRTlHTGJMMFNoa1FZcUI2MyUyQiUyRk5TSnNMbHNiNUVLVlNMTHM5T3olMkJ0NVVsSCUyQjAwTHdvVXFZd0VVVjdBJTNEJTNE; tncid=a732ead6-df87-46f0-8813-50f3295d7218; _ga_DVR9V968VW=GS2.1.s1764097199$o3$g0$t1764097202$j57$l0$h0; _ga_KVFYCX9ZTZ=GS2.1.s1764182687$o10$g1$t1764182814$j15$l0$h0; _ga=GA1.1.14977322.1756830969"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.fantacalcio.it/voti-fantacalcio-serie-a",
            "Cookie": COOKIE
        }
    
    print(f"⬇️  Inizio download per la stagione ID {SEASON_ID}...\n")

    for giornata in range(1, TOTAL_MATCHDAYS + 1):
        url = f"{BASE_URL}/{SEASON_ID}/{giornata}"
        filename = os.path.join(output_folder, f"Voti_Giornata_{giornata}.xlsx")

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                if len(response.content) > 1000:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print(f"✅ Giornata {giornata}: Scaricata correttamente.")
                else:
                    print(f"⚠️  Giornata {giornata}: File vuoto (forse non giocata).")
            elif response.status_code == 401:
                print(f"❌ Giornata {giornata}: Errore 401. I cookie potrebbero essere scaduti o errati.")
                break
            elif response.status_code == 404:
                print(f"❌ Giornata {giornata}: Non ancora disponibile.")
            else:
                print(f"❌ Giornata {giornata}: Errore HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️  Errore: {e}")
        time.sleep(1)
    url_quotazioni = f"https://www.fantacalcio.it/api/v1/Excel/prices/{SEASON_ID}/1"
    filename_quotazioni = os.path.join(output_folder, "Fantavalori.xlsx")

    try:
        response = requests.get(url_quotazioni, headers=headers)
        
        if response.status_code == 200:
            if len(response.content) > 1000:
                with open(filename_quotazioni, "wb") as f:
                    f.write(response.content)
                print(f"✅ Quotazioni scaricate correttamente in: {filename_quotazioni}")
            else:
                print(f"⚠️  File quotazioni vuoto o troppo piccolo.")
        elif response.status_code == 404:
            print(f"❌ File quotazioni non trovato (404).")
        else:
            print(f"❌ Errore download quotazioni: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️  Errore nel download delle quotazioni: {e}")

    print("\n🏁 Processo terminato.")


def clean_fantacalcio_data(df, vote_column_name):
    """
    Pulisce il DataFrame grezzo di Fantacalcio.it, estrae i Bonus 
    e calcola il Fantavoto.
    """
    
    mask_team = df.iloc[:, 1].isna()
    df['Squadra'] = df.iloc[:, 0].where(mask_team).ffill()
    
    col_map = {
        0: 'Cod', 1: 'Ruolo', 2: 'Nome', 3: 'Voto',
        4: 'Gf', 5: 'Gs', 6: 'Rp', 7: 'Rs', 8: 'Rf', 9: 'Au', 
        10: 'Amm', 11: 'Esp', 12: 'Ass'
    }
    df = df.rename(columns=col_map)

    df = df[pd.to_numeric(df['Cod'], errors='coerce').notnull()]
    df['Cod'] = df['Cod'].astype(int)

    df = df.rename(columns={'Voto': vote_column_name})

    df['Ruolo'] = df['Ruolo'].astype(str).str.strip().str.upper()
    df = df[df['Ruolo'].isin(['D', 'C', 'A'])]
    
    df['Squadra'] = df['Squadra'].astype(str).str.strip()
    
    bonus_cols = ['Gf', 'Gs', 'Rp', 'Rs', 'Rf', 'Au', 'Amm', 'Esp', 'Ass']
    for col in bonus_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Rigori_Sbagliati'] = df['Rs'] + df['Rp']
    
    voto_numeric = df[vote_column_name].astype(str).str.replace('*', '', regex=False)
    voto_numeric = pd.to_numeric(voto_numeric, errors='coerce')
    
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
    try:
        df_fanta_raw = pd.read_excel(file_path, sheet_name="Fantacalcio", skiprows=4, header=None)
        df_stat_raw = pd.read_excel(file_path, sheet_name="Statistico", skiprows=4, header=None)
        df_ita_raw = pd.read_excel(file_path, sheet_name="Italia", skiprows=4, header=None)

        df_fanta = clean_fantacalcio_data(df_fanta_raw, 'Voto_Fantacalcio')
        df_stat = clean_fantacalcio_data(df_stat_raw, 'Voto_Statistico')
        df_ita = clean_fantacalcio_data(df_ita_raw, 'Voto_Italia')
        
        if df_fanta is None or df_fanta.empty: return None

        df_final = df_fanta.merge(df_stat[['Cod', 'Voto_Statistico']], on='Cod', how='left')
        df_final = df_final.merge(df_ita[['Cod', 'Voto_Italia']], on='Cod', how='left')

        df_final['sv'] = np.where(df_final['Voto_Fantacalcio'].astype(str).str.endswith('*'), 1, 0)

        cols_voti = ["Fantacalcio", "Statistico", "Italia"]
        for col in cols_voti:
            column = f"Voto_{col}"

            df_final[column] = df_final[column].astype(str).str.replace('*', '', regex=False)
            df_final[column] = pd.to_numeric(df_final[column], errors='coerce')

        return df_final
    
    except Exception as e:
        print(f"Errore file {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()


def integra_quotazioni_per_codice(df_fanta, path_quotazioni):
    """
    Carica il file Excel delle quotazioni (fogli 'Tutti' e 'Ceduti')
    e unisce la colonna 'Qt.I' al dataframe dei voti usando il 'Cod' (Id).
    """
    
    try:
        df_tutti = pd.read_excel(path_quotazioni, sheet_name='Tutti', header=1)
        
        try:
            df_ceduti = pd.read_excel(path_quotazioni, sheet_name='Ceduti', header=1)
            df_valori = pd.concat([df_tutti, df_ceduti], ignore_index=True)
        except:
            print("⚠️ Foglio 'Ceduti' non trovato, uso solo 'Tutti'.")
            df_valori = df_tutti

        df_valori.columns = df_valori.columns.str.strip()
        
        col_qt_i = None
        for col in df_valori.columns:
            if "Qt" in col and "I" in col: 
                col_qt_i = col
                break
        
        if not col_qt_i:
            print("❌ Errore: Colonna 'Qt.I' non trovata nel file valori.")
            return df_fanta

        df_valori_clean = df_valori[['Id', col_qt_i]].copy()
        df_valori_clean = df_valori_clean.rename(columns={'Id': 'Cod', col_qt_i: 'Qt.I'})
        
        df_valori_clean['Cod'] = pd.to_numeric(df_valori_clean['Cod'], errors='coerce')
        df_valori_clean = df_valori_clean.dropna(subset=['Cod'])
        df_valori_clean['Cod'] = df_valori_clean['Cod'].astype(int)
        
        df_valori_clean = df_valori_clean.drop_duplicates(subset=['Cod'])

        df_merged = df_fanta.merge(df_valori_clean, on='Cod', how='left')
        
        df_merged['Qt.I'] = df_merged['Qt.I'].fillna(1).astype(int)
        
        print(f"✅ Quotazioni unite con successo! Righe totali: {len(df_merged)}")
        return df_merged

    except Exception as e:
        print(f"❌ Errore critico nell'integrazione quotazioni: {e}")
        return df_fanta


def extract_fanta(folder, season):
    base_folder = os.path.join(folder, f"voti_fanta_{season}") 
    os.makedirs(base_folder, exist_ok=True)
    fanta_scraper(base_folder, season)
    all_dfs = []

    print(f"Estrazione dati da: {base_folder}")

    for i in range(1, TOTAL_MATCHDAYS+1):
        file_path = os.path.join(base_folder, f'Voti_Giornata_{i}.xlsx')
        
        if os.path.exists(file_path):
            df_giornata = single_file(file_path)
            
            if not df_giornata.empty:
                df_giornata["id_giornata"] = i
                all_dfs.append(df_giornata)
    
    if all_dfs:
        total_df = pd.concat(all_dfs, ignore_index=True)

        file_quotazioni = os.path.join(base_folder, "Fantavalori.xlsx")
        final_df = integra_quotazioni_per_codice(total_df, file_quotazioni)
        final_df.to_excel(os.path.join(folder, "Fanta_Total.xlsx"),index=False)
        return final_df
    else:
        return pd.DataFrame()