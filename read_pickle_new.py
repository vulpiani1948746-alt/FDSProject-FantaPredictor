import pandas as pd
import os

# 1. La tua cartella principale (Base)
FOLDER = "C:/Users/New/Downloads/FDSProject-FantaPredictor-main/FDSProject-FantaPredictor-main/final_data"

# 2. La stagione che vuoi leggere (Basta cambiare qui per leggere un'altra annata!)
SEASON = "2024-2025"

# 3. Costruisci il percorso in automatico
# Diventa: .../final_data/season2024-2025/teams.pkl
percorso_file = os.path.join(FOLDER, f"season{SEASON}", "FinalDF_complete.pkl")

# 4. Leggi
df = pd.read_pickle(percorso_file)
print(f"Sto leggendo i dati della stagione: {SEASON}")
print(df.head())