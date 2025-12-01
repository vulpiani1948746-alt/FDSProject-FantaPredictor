import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from scaler import create_scaler
from feature_engineering import merge_players_with_teams
pd.set_option('display.max_rows', None)


FOLDER = "C:/Users/New/Downloads/FDSProject-FantaPredictor-main/FDSProject-FantaPredictor-main/final_data"  # Put your folder that will contain all the data
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 

# --- PARAMETRI ---
n_past = 5        # Guardiamo le ultime 5 partite
m_future = 3      # Prevediamo le prossime 3
threshold_nan = 3 # Se mancano 3 o più voti nello storico, scartiamo

player_cols = [
    'Voto_Fantacalcio', 'Player_Goals', 'Ast', 'Shots', 'SoT', 
    'Player_xG', 'Min', 'Touches', 'Voto_Diff', 'Age', 'Value', 'Min', 'Autogoal', 'Missed_Pen',
    'PK', 'CardY', 'CardR', 'Tkl', 'Int', 'Blocks', 'Player_xAST', 'SCA', 'GCA', 'Cmp%_x', 'PrgP',
    'PrgC', 'Cross', 'Succ', 'Fantavoto', 'Player_xG_Overperf', 'Shot_Accuracy', 'Role_A', 'Role_C',
    'Role_D', 'Voto_Diff', 'Avg_Voto_Fantacalcio', 'Avg_Fantavoto', 'Avg_Voto_Diff'
]

team_past_cols = [
    'Team_Goals', 'Opp_Goals', 'Team_xG', 'Opp_xG', 'Match_Result', 
    'is_empty', 'Match_Goal_Diff', 'xG_Diff'
]

team_future_cols = [
    'Pre_Match_Strength', 'Opponent_Strength', 'Strength_Diff', 'home', 
    'Pred_Team_Goals', 'Pred_Team_Conceded', 'Pred_Goal_Diff'
]

target_col = 'Fantavoto'

cols_to_scale = player_cols + team_past_cols + team_future_cols


def prepare_data_dual_input(player_df, team_df, scaler=None, fit=False):
    """
    Unisce i dataframe e scala i dati usando uno scaler pre-esistente.
    Restituisce il dataframe unito e lo scaler utilizzato.
    """
    merged_df = merge_players_with_teams(player_df, team_df)
    
    merged_df = merged_df.sort_values(by=['Player_id', 'Week_id'])
    
    if scaler is not None:
        merged_df = scaler.transform(merged_df)
        
    return merged_df, scaler


def prepare_all_data():
    all_data = []
    global_scaler = create_scaler(player_cols, team_past_cols, team_future_cols, SEASONS, FOLDER)

    print("Scaler creato. Inizio processamento stagioni...")

    for season in SEASONS:
        df_players = pd.read_pickle(os.path.join(FOLDER, f"season{season}", "FinalDF_player.pkl"))
        df_teams = pd.read_pickle(os.path.join(FOLDER, f"season{season}", "teams.pkl"))
        
        cols_preds = ['Pred_Team_Goals', 'Pred_Team_Conceded', 'Pred_Goal_Diff']
        for col in cols_preds:
            if col in df_teams.columns:
                # Se manca il dato, usiamo la media della colonna
                media = df_teams[col].mean()
                df_teams[col] = df_teams[col].fillna(media)
            else:
                # Se la colonna non esiste proprio (vecchi file), metti 0
                df_teams[col] = 0.0

        df_final_scaled, _ = prepare_data_dual_input(
            df_players, 
            df_teams, 
            scaler=global_scaler,
            fit=False
        )

        all_data.append(df_final_scaled)
    
    lower_bound, upper_bound = global_scaler.bounds['Fantavoto']
    range_voti = upper_bound - lower_bound

    return all_data, range_voti
    

def create_sequences_dual(df, n_past, m_future):
    """
    Crea i tensori X e y per la rete neurale Dual-Input.
    Gestisce l'imputazione dei NaN e lo scarto delle sequenze non valide.
    """
    X_past_list = []    # Input LSTM: (samples, n_past, n_features_past)
    X_fut_list = []     # Input Dense: (samples, m_future * n_features_fut)
    y_list = []         # Target: (samples, )
    
    # Uniamo le feature per lo storico
    past_features = player_cols + team_past_cols 
    future_features = team_future_cols           
    
    # Indice della colonna 'Voto' relativo al blocco 'player_cols'
    # Serve per controllare i NaN specifici del voto
    idx_voto = player_cols.index('Fantavoto')
    
    # Raggruppiamo per Giocatore
    for player_id, group in df.groupby('Player_id'):
        
        # Convertiamo in matrici Numpy per velocità (pandas è lento nei loop)
        data_past = group[past_features].values
        data_future_ctx = group[future_features].values
        data_target = group[target_col].values
        
        n_rows = len(group)
        
        # Se non ho abbastanza partite per fare una sequenza, salto il giocatore
        if n_rows < n_past + m_future:
            continue
            
        # Sliding Window Loop
        for i in range(n_past, n_rows - m_future + 1):
            
            # --- A. INPUT STORICO (LSTM) ---
            # Prendo le n_past righe precedenti
            window_past = data_past[i-n_past : i].copy()
            
            # --- LOGICA GESTIONE NaN ---
            # Guardiamo solo la sottomatrice relativa alle colonne del Giocatore
            # (Le colonne Team non dovrebbero avere NaN)
            player_sub_window = window_past[:, :len(player_cols)]
            
            # Controlliamo i NaN nella colonna Voto
            voti_window = player_sub_window[:, idx_voto]
            num_nan = np.isnan(voti_window).sum()
            
            # REGOLA 1: Se mancano troppi voti (>= 3 su 5), scarta la sequenza.
            if num_nan >= threshold_nan: 
                continue 
            
            # REGOLA 2: Se mancano pochi voti (< 3), riempili con la media locale.
            if num_nan > 0:
                # Calcola media colonna per colonna ignorando i NaN
                col_means = np.nanmean(player_sub_window, axis=0)
                
                # Se una colonna è TUTTA NaN, la media è NaN -> sostituisci con 0 per sicurezza
                col_means = np.nan_to_num(col_means, nan=0.0)
                
                # Trova dove sono i NaN
                inds = np.where(np.isnan(player_sub_window))
                
                # Sostituisci i NaN con la media della rispettiva colonna
                player_sub_window[inds] = np.take(col_means, inds[1])
                
                # Reinserisci la sottomatrice corretta nella finestra principale
                window_past[:, :len(player_cols)] = player_sub_window
            
            # --- B. INPUT FUTURO (Dense) ---
            # Prendo le m_future righe successive (SOLO colonne Strength/Home)
            # Flattening: trasformiamo da (3, 4) a vettore lungo (12,)
            window_future = data_future_ctx[i : i+m_future].flatten()
            
            # --- C. TARGET (y) ---
            # Prendo i voti futuri reali
            targets_future = data_target[i : i+m_future]
            
            # REGOLA 3: Se nel futuro manca anche solo un voto (non ha giocato),
            # non possiamo usare questo esempio per il training. Scarta.
            if np.isnan(targets_future).any():
                continue
            
            X_past_list.append(window_past)
            X_fut_list.append(window_future)
            y_list.append(np.mean(targets_future)) # Target è la media dei voti futuri

    return np.array(X_past_list), np.array(X_fut_list), np.array(y_list)


def generate_all_sequences(all_scaled_dfs, n_past, m_future):
    """
    Itera su tutti i dataframe scalati e genera le sequenze LSTM/Dense per ogni stagione.
    Restituisce una lista di tuple: [(X_past, X_fut, y), (X_past, X_fut, y), ...]
    """
    all_sequences = []
    
    print(f"Inizio generazione sequenze per {len(all_scaled_dfs)} stagioni...")

    for i, df in enumerate(all_scaled_dfs):
        # Chiama la tua funzione create_sequences_dual su ogni singolo dataframe
        X_p, X_f, y = create_sequences_dual(df, n_past, m_future)
        
        # Salva la tupla di risultati
        all_sequences.append((X_p, X_f, y))
        
        print(f"Stagione {i}: generati {len(y)} campioni.")

    return all_sequences


def prepare_sequences():
    tuple1 = prepare_all_data()
    scaled_dfs = tuple1[0]
    range_voti = tuple1[1]

    sequences_list = generate_all_sequences(scaled_dfs, n_past=5, m_future=3)

    X_train_past = np.concatenate([s[0] for s in sequences_list[:5]])
    X_train_fut  = np.concatenate([s[1] for s in sequences_list[:5]])
    y_train      = np.concatenate([s[2] for s in sequences_list[:5]])
    train = [X_train_past, X_train_fut, y_train]

    X_val_past   = sequences_list[5][0]
    X_val_fut    = sequences_list[5][1]
    y_val        = sequences_list[5][2]
    val = [X_val_past, X_val_fut, y_val]

    X_test_past  = sequences_list[6][0]
    X_test_fut   = sequences_list[6][1]
    y_test       = sequences_list[6][2]
    test = [X_test_past, X_test_fut, y_test]

    return train, val, test, range_voti
    
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model(shape_past, shape_future):
    # --- RAMO 1: LSTM (Analizza lo storico: Forma fisica, gol recenti...) ---
    # shape_past sarà (n_past, n_features_past)
    input_past = Input(shape=shape_past, name='Input_Storico_LSTM')
    
    # LSTM Layer
    # 64 unità: abbastanza potente per capire pattern complessi
    x = LSTM(64, return_sequences=False, activation='tanh')(input_past)
    x = BatchNormalization()(x) # Stabilizza l'apprendimento
    x = Dropout(0.3)(x)         # Spegne il 30% dei neuroni per evitare che la rete "impari a memoria"
    
    # --- RAMO 2: DENSE (Analizza il futuro: Contro chi gioco?) ---
    # shape_future sarà (m_future * n_features_fut,)
    input_future = Input(shape=shape_future, name='Input_Contesto_Futuro')
    
    # Dense Layer semplice
    y = Dense(32, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)
    
    # --- UNIONE (CONCATENATE) ---
    # Uniamo il vettore di stato della LSTM con il vettore del contesto futuro
    combined = Concatenate()([x, y])
    
    # --- CERVELLO CENTRALE (Interpretazione) ---
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.2)(z)
    z = Dense(32, activation='relu')(z)
    
    # --- OUTPUT (Regressione) ---
    # 1 solo neurone, attivazione lineare (perché prevediamo un voto, es. 6.5)
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    # Creazione Modello
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    # Compilazione
    # Loss = MAE (Mean Absolute Error): Vogliamo minimizzare l'errore assoluto (es. sbagliare di 0.3)
    # Optimizer = Adam con learning rate standard
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mse'])
    
    return model

# --- ISTANZIAMO IL MODELLO ---
# Prendiamo le dimensioni direttamente dai tuoi dati di train
lista = prepare_sequences()
range_voti = lista[3]
print(range_voti)
train = lista[0]
X_train_past = train[0]
X_train_fut = train[1]
y_train = train[2]

shape_p = (X_train_past.shape[1], X_train_past.shape[2]) # (5, n_features)
shape_f = (X_train_fut.shape[1],)                        # (n_features_fut,)

model = build_model(shape_p, shape_f)
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    # 1. EarlyStopping: Se il modello non migliora sul Validation Set per 15 epoche, fermati.
    # restore_best_weights=True è fondamentale: alla fine ti ridà il modello migliore, non l'ultimo.
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    
    # 2. ReduceLROnPlateau: Se non migliora per 5 epoche, rallenta il learning rate (fai passi più piccoli).
    # Aiuta a scendere nel minimo globale dell'errore.
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
]


test = lista[2]
X_test_past = test[0]
X_test_fut = test[1]
y_test = test[2]

"""
print("--- INIZIO TRAINING ---")

val = lista[1]
X_val_past = val[0]
X_val_fut = val[1]
y_val = val[2]

history = model.fit(
    # LISTA DEGLI INPUT (L'ordine deve corrispondere ai layer Input definiti in build_model)
    x=[X_train_past, X_train_fut], 
    
    # TARGET
    y=y_train,
    
    # VALIDATION SET (Fondamentale per capire se sta imparando davvero)
    validation_data=([X_val_past, X_val_fut], y_val),
    
    epochs=150,          # Mettiamo tante epoche, tanto EarlyStopping fermerà prima
    batch_size=64,       # 64 campioni alla volta. Se hai poca RAM, metti 32.
    callbacks=callbacks,
    verbose=1            # Mostra la barra di progresso
)

print("--- TRAINING COMPLETATO ---")

import matplotlib.pyplot as plt


# 1. Grafico dell'Errore (Loss)
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Train Loss (MAE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MAE)')
plt.title('Andamento Errore durante il Training')
plt.xlabel('Epoche')
plt.ylabel('Errore Medio Assoluto (Voto)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 2. Test Finale sulla Stagione 6 (Mai vista dal modello)
loss, mse = model.evaluate([X_test_past, X_test_fut], y_test)
print(f"\nMAE finale sul Test Set (Stagione 7): {loss:.4f}")

print(f"Ampiezza del range: {range_voti}")
    # Calcoliamo l'errore reale
real_mae = loss * range_voti
print(f"MAE Reale: {real_mae:.4f}")"""




from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Trova l'indice della colonna 'Voto_Fantacalcio' nei tuoi dati
# (Dobbiamo sapere quale "fetta" del tensore X guardare)
idx_voto = player_cols.index('Fantavoto')

# 2. Estrai solo i voti passati dal Test Set
# X_test_past ha forma (n_samples, 5, n_features)
# Noi prendiamo (n_samples, 5) -> Solo i voti delle 5 giornate precedenti
past_votes_scaled = X_test_past[:, :, idx_voto]

# 3. Calcola la media semplice (SMA) lungo l'asse temporale (axis=1)
# Questa è la nostra "Baseline Prediction"
baseline_preds_scaled = np.mean(past_votes_scaled, axis=1)

# 4. Calcola l'errore sui dati SCALATI (per confronto diretto col tuo 0.081)
baseline_mae_scaled = mean_absolute_error(y_test, baseline_preds_scaled) * 7.5

print(f"--- CONFRONTO ERRORI (SCALATI) ---")
print(f"MAE Baseline (Media 5):  {baseline_mae_scaled:.4f}")
