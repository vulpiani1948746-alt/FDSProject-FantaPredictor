import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from scaler import create_scaler, RobustMinMaxScaler
from feature_engineering import merge_players_with_teams
from baseline import baseline_linear_regression
from scipy.stats import spearmanr
pd.set_option('display.max_rows', None)


FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Put your folder that will contain all the data
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 

N_PAST = 5
M_FUTURE = 1
SCALER_TYPE = "robust"

player_cols_new = [
    'Voto_Fantacalcio', 'Player_Goals', 'Ast', 'Shots', 'SoT', 
    'Min', 'Touches', 'Voto_Diff', 'Age', 'Value', 'Min',
    'PK', 'CardY', 'Blocks', 'Player_xAST', 'SCA', 'GCA', 'Cmp%_x', 'PrgP',
    'PrgC', 'Cross', 'Succ', 'Fantavoto', 'Player_xG_Overperf', 'Shot_Accuracy', 'Role_A', 'Role_C',
    'Avg_Voto_Fantacalcio', 'Avg_Fantavoto', 'Avg_Voto_Diff'
]

team_past_cols_new = [
    'Team_Goals', 'Team_xG', 'Opp_xG', 'Match_Result', 
    'Match_Goal_Diff', 'xG_Diff'
]

team_future_cols = [
    'Pre_Match_Strength', 'Opponent_Strength', 'Strength_Diff', 'home'
]

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


def generate_all_sequences(all_scaled_dfs, n_past=N_PAST, m_future=M_FUTURE):
    all_sequences = []
    print(f"Inizio generazione sequenze (con AUGMENTATION) per {len(all_scaled_dfs)} stagioni...")

    for i, df in enumerate(all_scaled_dfs):
        # USIAMO LA FUNZIONE AUGMENTED
        X_p, X_f, y = create_sequences_dual_augmented(df, n_past, m_future)
        
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
    

def build_best_model_lstm(shape_past, shape_future):
    # --- 1. INPUT ---
    input_past = Input(shape=shape_past, name='Input_Storico_LSTM')
    input_future = Input(shape=shape_future, name='Input_Contesto_Futuro')
    
    # --- 2. RAMO LSTM (Vincitore: 64 units, Dropout 0.4) ---
    x = LSTM(96, return_sequences=False, activation='tanh')(input_past)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x) # <-- Dropout alto qui
    
    # --- 3. RAMO CONTESTO (Vincitore: 16 units, Dropout 0.4) ---
    y = Dense(16, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(0.4)(y) # <-- E qui
    
    # --- 4. UNIONE ---
    combined = Concatenate()([x, y])
    
    # --- 5. LAYER FINALE 1 (Vincitore: 128 units, Dropout 0.4) ---
    z = Dense(96, activation='relu')(combined)
    z = Dropout(0.4)(z) # <-- E qui
    
    # --- 6. LAYER EXTRA (Vincitore: True) ---
    # Il tuner aggiungeva un layer grande la metà del precedente
    z = Dense(96, activation='relu')(z) # 128 // 2 = 64
    z = Dropout(0.4)(z) # <-- E anche qui
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    # --- 7. COMPILAZIONE (Vincitori: LR 0.001, Cosine Wgt 5.0) ---
    final_loss = get_ranking_loss(weight_mse=1.0, weight_cos=10.0)
    
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                  loss=final_loss, 
                  metrics=['mae', 'mse'])
    
    return model


def train_and_test_best_config(plot_graphs=False):
    print("--- AVVIO TRAINING MODELLO OTTIMIZZATO (NO EMBEDDING) ---")
    
    lista = prepare_sequences(SCALER_TYPE, N_PAST) 
    denorm_factor = lista[3]
    
    train = lista[0]
    X_train_past, X_train_fut, y_train = train[0], train[1], train[2]
    
    val = lista[1]
    X_val_past, X_val_fut, y_val = val[0], val[1], val[2]
    
    test = lista[2]
    X_test_past, X_test_fut, y_test = test[0], test[1], test[2]

    shape_p = (X_train_past.shape[1], X_train_past.shape[2])
    shape_f = (X_train_fut.shape[1],)
    
    model = build_best_model_lstm(shape_p, shape_f)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-5, verbose=1)
    ]

    history = model.fit(
        x=[X_train_past, X_train_fut], 
        y=y_train,
        validation_data=([X_val_past, X_val_fut], y_val),
        epochs=150,      
        batch_size=64,   
        callbacks=callbacks,
        verbose=1 
    )

    print("TRAINING COMPLETATO.")

    if plot_graphs:
        plot_diagnostics(history, model, X_test_past, X_test_fut, y_test, denorm_factor)

    print("\n--- VALUTAZIONE FINALE ---")
    preds = model.predict([X_test_past, X_test_fut], verbose=0).flatten()
    
    mae_real = mean_absolute_error(y_test, preds) * denorm_factor
    spearman_score, _ = spearmanr(y_test, preds)
    
    print(f"MAE Reale: {mae_real:.4f}")
    print(f"Spearman:  {spearman_score:.4f}")
    
    return model, mae_real



def plot_diagnostics(history, model, X_past, X_fut, y_true, denorm_factor, lower_bound=0):
    """
    Plotta la Loss e un confronto predizioni vs realtà su un campione.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', linewidth=2)
    ax1.set_title('Curva di Apprendimento (Loss)')
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('MAE (Errore Assoluto)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    n_samples = 30
    if len(y_true) < n_samples: n_samples = len(y_true)
    
    indices = np.random.choice(len(y_true), n_samples, replace=False)
    
    X_p_sample = X_past[indices]
    X_f_sample = X_fut[indices]
    y_true_sample = y_true[indices]
    
    preds_sample = model.predict([X_p_sample, X_f_sample], verbose=0).flatten()
    
    y_real = y_true_sample * denorm_factor + lower_bound
    preds_real = preds_sample * denorm_factor + lower_bound
    
    x_axis = np.arange(n_samples)
    ax2.plot(x_axis, y_real, marker='o', linestyle='-', color='grey', alpha=0.5, label='Voto Reale')
    ax2.scatter(x_axis, preds_real, color='red', s=100, marker='x', label='Predizione AI', zorder=10)
    ax2.vlines(x_axis, ymin=y_real, ymax=preds_real, colors='orange', alpha=0.4)
    
    ax2.set_title(f'Realtà vs Predizione (Campione di {n_samples})')
    ax2.set_ylabel('Voto Fantacalcio')
    ax2.set_xticks(x_axis)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def get_ranking_loss(weight_mse=1.0, weight_cos=5.0):
    """
    Loss Ibrida:
    - MSE: Per azzeccare il valore del voto (es. 6.5).
    - Cosine: Per azzeccare la direzione/ranking (es. A > B).
    """
    mse_fn = tf.keras.losses.MeanSquaredError()
    cosine_fn = tf.keras.losses.CosineSimilarity()

    def ranking_loss(y_true, y_pred):
        loss_mse = mse_fn(y_true, y_pred)
        
        loss_cos = cosine_fn(y_true, y_pred)
        
        return (weight_mse * loss_mse) + (weight_cos * loss_cos)

    return ranking_loss


def create_sequences_dual_augmented(df, n_past=N_PAST, m_future=M_FUTURE):
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
            
            # --- PREPARAZIONE DATI (Standard) ---
            window_past = data_past[i-n_past : i].copy()
            
            # Gestione NaN Player
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
            
            # Cleaning finale
            if np.isnan(window_past).any(): window_past = np.nan_to_num(window_past, nan=0.0)
            
            window_future = data_future_ctx[i : i+m_future].flatten()
            if np.isnan(window_future).any(): window_future = np.nan_to_num(window_future, nan=0.0)
            
            targets_future = data_target[i : i+m_future]
            if np.isnan(targets_future).any(): continue
            
            # Calcolo il target finale
            target_val = np.mean(targets_future)
            
            # --- LOGICA DI AUGMENTATION (Strategia 1) ---
            
            # 1. Aggiungiamo SEMPRE la sequenza originale (Caso base)
            X_past_list.append(window_past)
            X_fut_list.append(window_future)
            y_list.append(target_val)
            
            # 2. OVER-SAMPLING PER VOTI ALTI
            # Se il voto è un "Exploit" (> 10), lo facciamo vedere altre 3 volte alla rete
            if target_val >= 10.0:
                for _ in range(3): 
                    X_past_list.append(window_past)
                    X_fut_list.append(window_future)
                    y_list.append(target_val)
                    
            # Se il voto è "Ottimo" (8.0 - 9.99), lo facciamo vedere 1 volta extra
            elif target_val >= 8.0:
                X_past_list.append(window_past)
                X_fut_list.append(window_future)
                y_list.append(target_val)

    return np.array(X_past_list), np.array(X_fut_list), np.array(y_list)


if __name__ == "__main__":
    model, final_mae = train_and_test_best_config(plot_graphs=True)