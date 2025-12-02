import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
from scipy.stats import spearmanr
pd.set_option('display.max_rows', None)


FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Put your folder that will contain all the data
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 

N_PAST = 5
M_FUTURE = 1
SCALER_TYPE = "robust"

player_cols = [
    'Voto_Fantacalcio', 'Player_Goals', 'Ast', 'Shots', 'SoT', 
    'Min', 'Touches', 'Voto_Diff', 'Age', 'Value', 'Min',
    'PK', 'CardY', 'Blocks', 'Player_xAST', 'SCA', 'GCA', 'Cmp%_x', 'PrgP',
    'PrgC', 'Cross', 'Succ', 'Fantavoto', 'Player_xG_Overperf', 'Shot_Accuracy', 'Role_A', 'Role_C',
    'Avg_Voto_Fantacalcio', 'Avg_Fantavoto', 'Avg_Voto_Diff'
]

team_past_cols = [
    'Team_Goals', 'Team_xG', 'Opp_xG', 'Match_Result', 
    'Match_Goal_Diff', 'xG_Diff'
]

team_future_cols = [
    'Pre_Match_Strength', 'Opponent_Strength', 'Strength_Diff', 'home'
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


def prepare_all_data(scaler_type = SCALER_TYPE):
    all_data = []
    global_scaler = create_scaler(player_cols, team_past_cols, team_future_cols, scaler_type,  SEASONS, FOLDER)

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
        cols_to_scale = player_cols + team_past_cols + team_future_cols
        idx = cols_to_scale.index(target_col_name)
        denorm_factor = global_scaler.data_range_[idx]
        
    elif isinstance(global_scaler, StandardScaler):
        cols_to_scale = player_cols + team_past_cols + team_future_cols
        idx = cols_to_scale.index(target_col_name)
        denorm_factor = global_scaler.scale_[idx]

    
    return all_data, denorm_factor
    

def create_sequences_dual(df, n_past=N_PAST, m_future=M_FUTURE):
    X_past_list = []
    X_fut_list = []
    y_list = []
    threshold_nan = math.ceil(n_past/2)
    
    past_features = player_cols + team_past_cols 
    future_features = team_future_cols           
    
    idx_voto = player_cols.index('Fantavoto')
    
    for player_id, group in df.groupby('Player_id'):
        
        data_past = group[past_features].values
        data_future_ctx = group[future_features].values
        data_target = group[target_col].values
        
        n_rows = len(group)
        
        if n_rows < n_past + m_future:
            continue
            
        for i in range(n_past, n_rows - m_future + 1):
            
            window_past = data_past[i-n_past : i].copy()
            
            player_sub_window = window_past[:, :len(player_cols)]
            voti_window = player_sub_window[:, idx_voto]
            num_nan = np.isnan(voti_window).sum()
            
            if num_nan >= threshold_nan: continue 
            
            if num_nan > 0:
                col_means = np.nanmean(player_sub_window, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                inds = np.where(np.isnan(player_sub_window))
                player_sub_window[inds] = np.take(col_means, inds[1])
                window_past[:, :len(player_cols)] = player_sub_window
            
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
    

def build_model(shape_past, shape_future):
    input_past = Input(shape=shape_past, name='Input_Storico_LSTM')
    
    x = LSTM(64, return_sequences=False, activation='tanh')(input_past)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)         
    
    input_future = Input(shape=shape_future, name='Input_Contesto_Futuro')
    
    y = Dense(32, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(0.1)(y)
    
    combined = Concatenate()([x, y])
    
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.1)(z)
    z = Dense(32, activation='relu')(z)
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    custom_loss = get_ranking_loss(weight_mse=1.0, weight_cos=5.0)

    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mae', 'mse'])
    
    return model


def train_and_test(n_past=N_PAST, scaler_type=SCALER_TYPE, plot_graphs=False):
    print("START TRAINING -----")
    lista = prepare_sequences(scaler_type, n_past) 
    denorm_factor = lista[3]
    
    train = lista[0]
    X_train_past, X_train_fut, y_train = train[0], train[1], train[2]
    val = lista[1]
    X_val_past, X_val_fut, y_val = val[0], val[1], val[2]
    test = lista[2]
    X_test_past, X_test_fut, y_test = test[0], test[1], test[2]

    shape_p = (X_train_past.shape[1], X_train_past.shape[2])
    shape_f = (X_train_fut.shape[1],)                       
    model = build_model(shape_p, shape_f)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5, verbose=0)
    ]

    history = model.fit(
        x=[X_train_past, X_train_fut], y=y_train,
        validation_data=([X_val_past, X_val_fut], y_val),
        epochs=100, batch_size=64, callbacks=callbacks, verbose=0 
    )
    print("TRAINING ENDED --------")

    if plot_graphs:
        plot_diagnostics(history, model, X_test_past, X_test_fut, y_test, denorm_factor)

    lstm_preds = model.predict([X_test_past, X_test_fut], verbose=0).flatten()
    
    lstm_mae = mean_absolute_error(y_test, lstm_preds) * denorm_factor
    
    lstm_spearman, _ = spearmanr(y_test, lstm_preds)
    
    print(f"\n--- RISULTATI LSTM ---")
    print(f"MAE Reale: {lstm_mae:.4f}")
    print(f"Spearman:  {lstm_spearman:.4f}")
    
    return model, lstm_mae


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
    
    n_samples = 20
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

if __name__ == "__main__":
    train_and_test(plot_graphs=True)