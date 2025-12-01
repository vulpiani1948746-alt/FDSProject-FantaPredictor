import numpy as np
import os
import math
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from lstm_with_predictions import get_ranking_loss, plot_diagnostics
from lstm_with_predictions import N_PAST, M_FUTURE, SCALER_TYPE 
from feature_engineering import merge_players_with_teams
from scaler import create_scaler, RobustMinMaxScaler, MinMaxScaler, StandardScaler

FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Put your folder that will contain all the data
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 


player_cols_new = [
    'Voto_Fantacalcio', 'Player_Goals', 'Ast', 'Shots', 'SoT', 
    'Player_xG', 'Min', 'Touches', 'Voto_Diff', 'Min', 'Autogoal', 'Missed_Pen',
    'PK', 'CardR', 'Tkl', 'Int', 'Blocks', 'Cmp%_x',
    'PrgC', 'Cross', 'Succ', 'Fantavoto', 'Player_xG_Overperf', 'Shot_Accuracy', 'Role_A', 'Role_C',
    'Role_D', 'Avg_Voto_Fantacalcio', 'Avg_Fantavoto'
]

team_past_cols_new = [
    'Team_Goals', 'Opp_Goals', 'Team_xG', 
    'is_empty', 'Match_Goal_Diff', 'xG_Diff'
]

team_future_cols_new = [
    'Pre_Match_Strength', 'Opponent_Strength', 'Strength_Diff', 'home',
    'Pred_Team_Goals', 'Pred_Team_Conceded', 'Pred_Goal_Diff'
]

target_col = 'Fantavoto'

cols_to_scale = player_cols_new + team_past_cols_new + team_future_cols_new


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


def prepare_all_data(scaler_type = SCALER_TYPE):
    all_data = []
    global_scaler = create_scaler(player_cols_new, team_past_cols_new, team_future_cols_new, scaler_type,  SEASONS, FOLDER)

    print("Scaler creato. Inizio processamento stagioni...")

    for season in SEASONS:
        df_players = pd.read_pickle(os.path.join(FOLDER, f"season{season}", "FinalDF_player.pkl"))
        df_teams = pd.read_pickle(os.path.join(FOLDER, f"season{season}", "teams.pkl"))
        
        df_final_scaled, _ = prepare_data_dual_input(
            df_players, 
            df_teams, 
            scaler=global_scaler,
            fit=False
        )

        all_data.append(df_final_scaled)
    
    idx_target = -1
    target_col_name = 'Fantavoto'
    
    if isinstance(global_scaler, RobustMinMaxScaler):
        lower, upper = global_scaler.bounds[target_col_name]
        denorm_factor = upper - lower
        
    elif isinstance(global_scaler, MinMaxScaler):
        cols_to_scale = player_cols_new + team_past_cols_new + team_future_cols_new
        idx = cols_to_scale.index(target_col_name)
        denorm_factor = global_scaler.data_range_[idx]
        
    elif isinstance(global_scaler, StandardScaler):
        cols_to_scale = player_cols_new + team_past_cols_new + team_future_cols_new
        idx = cols_to_scale.index(target_col_name)
        denorm_factor = global_scaler.scale_[idx]

    
    return all_data, denorm_factor
    

def create_sequences_dual(df, n_past=N_PAST, m_future=M_FUTURE):
    X_past_list = []
    X_fut_list = []
    y_list = []
    threshold_nan = math.ceil(n_past/2)
    
    past_features = player_cols_new + team_past_cols_new 
    future_features = team_future_cols_new           
    
    idx_voto = player_cols_new.index('Fantavoto')
    
    for player_id, group in df.groupby('Player_id'):
        
        data_past = group[past_features].values
        data_future_ctx = group[future_features].values
        data_target = group[target_col].values
        
        n_rows = len(group)
        
        if n_rows < n_past + m_future:
            continue
            
        for i in range(n_past, n_rows - m_future + 1):
            
            window_past = data_past[i-n_past : i].copy()
            
            player_sub_window = window_past[:, :len(player_cols_new)]
            voti_window = player_sub_window[:, idx_voto]
            num_nan = np.isnan(voti_window).sum()
            
            if num_nan >= threshold_nan: continue 
            
            if num_nan > 0:
                col_means = np.nanmean(player_sub_window, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                inds = np.where(np.isnan(player_sub_window))
                player_sub_window[inds] = np.take(col_means, inds[1])
                window_past[:, :len(player_cols_new)] = player_sub_window
            
            if np.isnan(window_past).any():
                window_past = np.nan_to_num(window_past, nan=0.0)
            
            window_future = data_future_ctx[i : i+m_future].flatten()

            if np.isnan(window_future).any():
                window_future = np.nan_to_num(window_future, nan=0.0)
            
            targets_future = data_target[i : i+m_future]
            if np.isnan(targets_future).any(): continue
            
            X_past_list.append(window_past)
            X_fut_list.append(window_future)
            y_list.append(np.mean(targets_future))

    return np.array(X_past_list), np.array(X_fut_list), np.array(y_list)


def generate_all_sequences(all_scaled_dfs, n_past = N_PAST, m_future = M_FUTURE):
    """
    Itera su tutti i dataframe scalati e genera le sequenze LSTM/Dense per ogni stagione.
    Restituisce una lista di tuple: [(X_past, X_fut, y), (X_past, X_fut, y), ...]
    """
    all_sequences = []
    
    print(f"Inizio generazione sequenze per {len(all_scaled_dfs)} stagioni...")

    for i, df in enumerate(all_scaled_dfs):
        X_p, X_f, y = create_sequences_dual(df, n_past, m_future)
        
        all_sequences.append((X_p, X_f, y))
        
        print(f"Stagione {i}: generati {len(y)} campioni.")

    return all_sequences


def prepare_sequences(scaler_type = SCALER_TYPE, n_past = N_PAST, m_future = M_FUTURE):
    tuple1 = prepare_all_data(scaler_type)
    scaled_dfs = tuple1[0]
    range_voti = tuple1[1]

    sequences_list = generate_all_sequences(scaled_dfs, n_past = n_past, m_future = m_future)

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

def build_cnn_model(shape_past, shape_future):
    """
    Rete Convoluzionale 1D per serie temporali.
    Ottima per catturare pattern locali (es. picchi di forma nelle ultime 3 partite).
    """
    input_past = Input(shape=shape_past, name='Input_Storico_CNN')
    input_future = Input(shape=shape_future, name='Input_Contesto')
    
    # --- RAMO CNN (Analizza il passato) ---
    # Layer 1: Filtri per pattern di base
    # kernel_size=3: Guarda finestre di 3 partite alla volta
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_past)
    x = BatchNormalization()(x)
    
    # Layer 2: Filtri per pattern più complessi
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Pooling: Riassume l'intera sequenza temporale in un vettore di caratteristiche
    # GlobalAveragePooling è spesso meglio di Flatten per evitare overfitting
    x = GlobalAveragePooling1D()(x) 
    x = Dropout(0.2)(x)
    
    # --- RAMO CONTESTO (Dense) ---
    # Identico alla LSTM, processa i dati futuri (avversario, casa/fuori)
    y = Dense(32, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)
    
    # --- UNIONE ---
    combined = Concatenate()([x, y])
    
    # --- LAYER FINALE ---
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.1)(z)
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    # Usiamo la tua loss ibrida (MSE + Ranking)
    # Peso 5.0 sul Ranking è un buon punto di partenza per le CNN
    loss_ranking = get_ranking_loss(weight_mse=1.0, weight_cos=5.0)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss=loss_ranking, 
                  metrics=['mae', 'mse'])
    
    return model

# --- 2. FUNZIONE DI TRAINING E TEST ---
def train_and_test_cnn(plot_graphs=True):
    print("\n=== AVVIO TRAINING CNN MODEL ===")
    
    # 1. Preparazione Dati (Riutilizziamo la logica esistente)
    # prepare_sequences gestisce caricamento, scaling, id mapping, ecc.
    # Assumiamo restituisca: [train, val, test, denorm, num_players]
    # Se la tua prepare_sequences non usa embeddings, restituisce solo 4 cose (senza num_players)
    # Adattiamo l'unpacking dinamicamente:
    
    data_bundle = prepare_sequences(SCALER_TYPE, N_PAST, M_FUTURE)
    
    # Unpack intelligente
    train = data_bundle[0]
    val = data_bundle[1]
    test = data_bundle[2]
    denorm_factor = data_bundle[3]
    
    # Controlliamo se ci sono gli ID (len del train > 2) o no
    # La CNN di base NON usa embedding ID qui, quindi prendiamo solo X_past e X_fut
    X_train_p, X_train_f = train[0], train[1]
    y_train = train[-1] # L'ultimo è sempre il target
    
    X_val_p, X_val_f = val[0], val[1]
    y_val = val[-1]
    
    X_test_p, X_test_f = test[0], test[1]
    y_test = test[-1]

    print(f"Dati caricati. Train shape: {X_train_p.shape}")

    # 2. Costruzione Modello
    shape_p = (X_train_p.shape[1], X_train_p.shape[2])
    shape_f = (X_train_f.shape[1],)
    
    model = build_cnn_model(shape_p, shape_f)
    # model.summary() # Decommenta se vuoi vedere i layer

    # 3. Callbacks
    # Le CNN convergono più in fretta delle LSTM, ma usiamo pazienza alta per sicurezza
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-6, verbose=1)
    ]

    # 4. Fit
    history = model.fit(
        x=[X_train_p, X_train_f], 
        y=y_train,
        validation_data=([X_val_p, X_val_f], y_val),
        epochs=150,      
        batch_size=64,   
        callbacks=callbacks,
        verbose=1 
    )

    print("TRAINING CNN COMPLETATO.")

    # 5. Plotting
    if plot_graphs:
        # Nota: Assicurati che plot_diagnostics accetti 2 input, o adattala se ne aspetta 3
        # Se plot_diagnostics è quella standard a 2 input:
        plot_diagnostics(history, model, X_test_p, X_test_f, y_test, denorm_factor)

    # 6. Valutazione Finale
    print("\n--- RISULTATI CNN ---")
    preds = model.predict([X_test_p, X_test_f], verbose=0).flatten()
    
    mae_scaled = mean_absolute_error(y_test, preds)
    mae_real = mae_scaled * denorm_factor
    spearman, _ = spearmanr(y_test, preds)
    
    print(f"MAE Reale: {mae_real:.4f}")
    print(f"Spearman:  {spearman:.4f}")
    
    return model, mae_real

if __name__ == "__main__":
    model, final_mae = train_and_test_cnn(plot_graphs=True)