from preclean_data import initial_data
from scraper_fanta import extract_fanta
import pandas as pd
from fuzzywuzzy import process, fuzz
import unicodedata
import os

# 1. DIZIONARIO CORREZIONI MANUALI
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
        "Pablo Galdames Millán": "Galdames", # Genoa
        "Pepín": "Machin",                 # Monza (spesso MACHIN o PEPIN) - Controlla se è MACHIN
        "Ruan": "Tressoldi",        
        "Seydou Fini": "Fini",   
        "Triantafyllos Pasaridīs": "Pasalidis"

        # Aggiungi qui eventuali altri nomi se escono dal find_problems
    }

def transliterate_name(input_str):
    """
    Pulisce i nomi gestendo caratteri speciali non standard (ð, æ, ø, Đ).
    """
    if pd.isna(input_str): return ""
    
    s = str(input_str).lower().strip()
    
    # 1. Mappa manuale per i caratteri "ostinati"
    # Aggiungi qui altri caratteri se ne trovi
    replacements = {
        'ð': 'd',
        'æ': 'ae',  # Esempio: Kjær -> Kjaer (o Maehle)
        'ø': 'o',   # Esempio: Højlund -> Hojlund
        'đ': 'dj',  # Esempio: Đurić -> Djuric
        'Ł': 'L',
        'ę': 'e',
        'ł': 'l',
        'ī': 'i',
        'à': 'a'
    }
    
    for char, replacement in replacements.items():
        s = s.replace(char, replacement)

    # 2. Normalizzazione standard per accenti classici (è, ö, ù, ì)
    nfkd_form = unicodedata.normalize('NFKD', s)
    s = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        
    return s.strip()


def pulisci_nome_fanta(nome_fanta):
    """
    Trasforma 'DE ROON M.' in 'DE ROON'.
    Rimuove l'iniziale puntata alla fine se presente.
    """
    if pd.isna(nome_fanta): return ""
    parts = str(nome_fanta).strip().split()
    
    if len(parts) > 1 and (len(parts[-1]) <= 2 or parts[-1].endswith('.')):
        return " ".join(parts[:-1]).lower()
    
    return " ".join(parts).lower()


def prepare_dataframes(df_fanta, df_stats):

    df_stats['team_norm'] = df_stats['Team_Name'].str.lower().str.strip()
    df_fanta['team_norm'] = df_fanta['Squadra'].str.lower().str.strip()

    # Creiamo le colonne chiave per il raggruppamento
    df_stats['match_key'] = df_stats['team_norm'] + "_" + df_stats['Wk'].astype(str)
    df_fanta['match_key'] = df_fanta['team_norm'] + "_" + df_fanta['id_giornata'].astype(str)

    df_fanta['fanta_key'] = df_fanta['Nome'].apply(pulisci_nome_fanta).apply(transliterate_name)
    df_stats['stats_key'] = df_stats['Player'].apply(transliterate_name)

    return df_fanta, df_stats


def find_problems(df_matches, df_stats):
    indici_mancanti = df_stats.index.difference(df_matches['index_stats'])
    df_missing = df_stats.loc[indici_mancanti].copy()

    print(f"Total non-matched rows: {len(df_missing)}")
    print("-" * 50)

    nomi_unici_sbagliati = sorted(df_missing['Player'].unique())

    print(f"Players to fix ({len(nomi_unici_sbagliati)}):")
    for nome in nomi_unici_sbagliati:
        squadra = df_missing[df_missing['Player'] == nome]['Team_Name'].iloc[0]
        print(f"❌ {nome} ({squadra})")

    print("-" * 50)

    if len(df_missing) > 20:
        print("❌❌ WE NEED TO FIX THE REPLACEMENTS AND RE-RUN ❌❌")
        exit()

    else:
        print("We can ignore a small amount of mismatches!!")
        print("Let's merge the two dataframes!!")


def find_matches2(df_fanta, df_stats, replacements):
    """
    Cerca corrispondenze tra Stats e Fanta.
    PRIORITÀ 1: Dizionario manuale (per i casi impossibili da risolvere con fuzzy).
    PRIORITÀ 2: Fuzzy Logic (token_set_ratio).
    """

    matches = []
    
    # Raggruppiamo per chiave partita (Squadra + Giornata) per velocità e precisione
    groups = df_stats.groupby('match_key')

    for key, group_stats in groups:
        subset_fanta = df_fanta[df_fanta['match_key'] == key]
        
        # Se non ci sono dati fanta per questa squadra/giornata, saltiamo
        if subset_fanta.empty:
            continue
        
        # --- PREPARAZIONE DATI PER IL MATCHING ---
        
        # 1. Mappa per Fuzzy: {chiave_pulita: Nome_Originale}
        # Es: {'de roon': 'DE ROON M.'}
        fanta_candidates_map = dict(zip(subset_fanta['fanta_key'], subset_fanta['Nome']))
        choices = list(fanta_candidates_map.keys())
        
        # 2. Set per Manuale: Nomi originali disponibili in questa giornata
        # Serve per verificare che il giocatore "forzato" sia effettivamente in lista
        fanta_names_original_set = set(subset_fanta['Nome'].values)

        for idx, row in group_stats.iterrows():
            player_stats_name = row['Player']      # Nome originale Stats (es. "Daniel Bandeira")
            player_clean_name = row['stats_key']   # Nome pulito Stats (es. "daniel bandeira")
            
            found_name = None
            score = 0
            
            if player_stats_name in replacements:
                target_name = replacements[player_stats_name]
                
                # Verifichiamo che il giocatore target sia effettivamente nella lista Fanta di QUESTA partita
                # (es. Se Gaspar era squalificato o non in lista, non dobbiamo forzare il match)
                if target_name in fanta_names_original_set:
                    found_name = target_name
                    score = 100 # Forziamo il punteggio massimo
            
            if found_name is None:
                # Usa token_set_ratio che gestisce bene "De Roon" in "Marten De Roon"
                best_match_key, calc_score = process.extractOne(
                    player_clean_name, choices, scorer=fuzz.token_set_ratio
                )
                
                if calc_score > 75: # Soglia di sicurezza
                    found_name = fanta_candidates_map[best_match_key]
                    score = calc_score
            
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
    Unisce i DataFrame Stats e Fanta usando la tabella di decodifica df_matches.
    Mantiene SOLO le righe dove c'è stato un match (INNER JOIN).
    """
    
    # 1. Uniamo df_stats con i match trovati
    # Usiamo 'inner' join: chi non è in df_matches viene scartato automaticamente
    df_merged = df_stats.merge(
        df_matches, 
        left_index=True, 
        right_on='index_stats', 
        how='inner' 
    )
    
    # 2. Uniamo i dati del Fantacalcio completi
    # Usiamo la chiave composta (match_key) + il nome fanta trovato dal matching
    final_df = df_merged.merge(
        df_fanta,
        left_on=['match_key', 'fanta_name'], 
        right_on=['match_key', 'Nome'],      
        how='inner', # Scarta eventuali incongruenze residue
        suffixes=('_stats', '_fanta')
    )
    
    # 3. Pulizia colonne tecniche (drop di quelle usate solo per il calcolo)
    cols_to_drop = [
        'stats_key', 'fanta_key', 'match_key', 'index_stats', 
        'team_norm_stats', 'team_norm_fanta', 'giornata_join',
        'match_score', 'fanta_name'
    ]
    # Rimuoviamo solo quelle che esistono effettivamente
    final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns], errors='ignore')
    
    # Reset dell'indice per avere un dataframe pulito
    final_df = final_df.reset_index(drop=True)

    return final_df


def clean_merged(df):
    cols_to_drop = [
        "Player",
        "Wk",
        "Team_Name"
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
        'Qt.I': 'Value'   
    }
    df = df.rename(columns=rename_map)

    final_order = [
        'Player_id', 'Name', 'Team', 'Is_Home', 'Pos', 'Age', 'Value', 'Opponent', 'Week_id', 
        'Min', 'Player_Goals', 'Ast', 'Autogoal', 'Missed_Pen', 'PK', 'PKatt', 'Shots', 'SoT', 
        'CardY', 'CardR', 'Touches', 'Tkl', 'Int', 'Blocks', 
        'Player_xG', 'Player_npxG', 'Player_xAST', 
        'SCA', 'GCA', 'Cmp_x', 'Att_x', 'Cmp%_x', 'PrgP', 'Carries', 'PrgC', 
        'Att_x.1', 'Succ', 'game_id', 
        'Team_Goals', 'Opp_Goals', 'Team_xG', 'Opp_xG', 
        'Voto_Fantacalcio', 'Voto_Statistico', 'Voto_Italia', 'sv', 'Fantavoto'
    ]
    df = df[final_order]

    return df


def return_final_dataframe(season, folder, league = "Serie-A"):

    df_teams, df_stats = initial_data(season, folder, league)
    df_fanta = pd.read_pickle(os.path.join(folder, "Fantacalcio.pkl"))

    df_fanta, df_stats = prepare_dataframes(df_fanta, df_stats)

    print("Find Matches between the two dataframes...")
    df_matches = find_matches2(df_fanta, df_stats, replacements)

    find_problems(df_matches, df_stats)


    df_final = merge_dataframes(df_stats, df_fanta, df_matches)
    print(f"✅ Merge completed! Final Rows: {len(df_final)}")
    df_final = clean_merged(df_final)
    
    return df_final

