import pandas as pd
import numpy as np
from preclean_data import initial_data
import os

def calculate_team_strength(df):
    # 1. ORDINE E COPIA
    df = df.sort_values(['Team_Name', 'Wk']).copy()

    # --- FASE A: CALCOLO PRESTAZIONE PARTITA SINGOLA ---
    
    # Punti base (3 vittoria, 1 pareggio) per il calcolo
    conditions = [
        (df['Goals'] > df['Goals_Ag']), 
        (df['Goals'] == df['Goals_Ag'])
    ]
    # Punti Reali (servono per il PPG)
    df['Raw_Points'] = np.select(conditions, [3, 1], default=0)
    
    # Punti Aggiustati (servono per la Forma con bonus trasferta)
    difficulty_multiplier = np.where(df['home'] == 0, 1.2, 1.0)
    df['Adjusted_Points'] = df['Raw_Points'] * difficulty_multiplier
    
    # xG Ratio
    df['xG_Ratio'] = df['xG'] / (df['xG'] + df['xG_Ag'] + 0.01)

    # --- FASE B: CALCOLO FORMA RECENTE (EMA) ---
    # Identico a prima: focus sulle ultime 5 partite
    
    df['Form_Points_EMA'] = df.groupby('Team_Name')['Adjusted_Points'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
    df['Form_xG_EMA'] = df.groupby('Team_Name')['xG_Ratio'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
    
    # Score Forma (0-100)
    df['Form_Score'] = ((df['Form_Points_EMA'] / 3.6) * 0.60 + (df['Form_xG_EMA'] * 0.40)) * 100

    # --- FASE C: CALCOLO STATUS (PPG STAGIONALE) - MODIFICATO ---
    
    # Calcoliamo la media punti cumulativa (Expanding Mean)
    # Expanding significa: alla giornata 10, fa la media della 1..10
    df['Season_PPG_Current'] = df.groupby('Team_Name')['Raw_Points'].transform(lambda x: x.expanding().mean())
    
    # IMPORTANTE: Shiftiamo di 1.
    # Il PPG che definisce lo "Status" per la partita di oggi è quello maturato fino a IERI.
    df['Pre_Match_PPG'] = df.groupby('Team_Name')['Season_PPG_Current'].shift(1)
    
    # Gestione valori nulli (Prima giornata):
    # Usiamo 1.35 che è circa la media punti globale nel calcio (una squadra media fa ~1.35 pti a partita)
    df['Pre_Match_PPG'] = df['Pre_Match_PPG'].fillna(1.35)
    
    # Normalizzazione 0-100
    # Dividiamo per 3.0 (il massimo teorico di punti). 
    # Esempio: PPG 2.4 (Scudetto) -> Score 80. PPG 0.9 (Retrocessione) -> Score 30.
    df['Status_Score'] = (df['Pre_Match_PPG'] / 3.0) * 100

    # --- FASE D: PESO DINAMICO (0.10 -> 0.35) ---
    
    max_wk = df['Wk'].max()
    df['Season_Progress'] = df['Wk'] / max_wk
    
    # Il peso dello "Status" cresce man mano che il PPG diventa statisticamente significativo
    df['Weight_Status'] = (df['Season_Progress'] * 0.5).clip(0.1, 0.5)
    df['Weight_Form'] = 1 - df['Weight_Status']

    # --- FASE E: CALCOLO SCORE FINALE ---
    
    df['Final_Strength_Raw'] = (df['Form_Score'] * df['Weight_Form']) + \
                               (df['Status_Score'] * df['Weight_Status'])

    # --- FASE F: PRE-MATCH SHIFT FINALE ---
    
    # Spostiamo il risultato finale alla riga successiva per evitare Data Leakage
    df['Pre_Match_Strength'] = df.groupby('Team_Name')['Final_Strength_Raw'].shift(1)
    
    # Valore di default per la prima partita (50.0 è neutro)
    df['Pre_Match_Strength'] = df['Pre_Match_Strength'].fillna(50.0)

    # --- PULIZIA ---
    cols_to_drop = ['Raw_Points', 'Adjusted_Points', 'xG_Ratio', 'Form_Points_EMA', 'Form_xG_EMA', 
                    'Form_Score', 'Season_PPG_Current', 'Pre_Match_PPG', 'Status_Score', 
                    'Season_Progress', 'Weight_Status', 'Weight_Form']
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df
