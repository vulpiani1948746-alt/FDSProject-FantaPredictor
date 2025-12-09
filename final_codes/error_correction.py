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
from scipy.stats import spearmanr
from lstm_with_predictions import plot_diagnostics, prepare_data_dual_input, prepare_all_data, N_PAST, SCALER_TYPE
pd.set_option('display.max_rows', None)


FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Put your folder that will contain all the data
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 

M_FUTURE = 1

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

team_past_cols_new = [
    'Team_Goals', 'Opp_Goals', 'Team_xG', 'Opp_xG', 'Match_Result', 
    'is_empty', 'Match_Goal_Diff', 'xG_Diff'
]

team_future_cols_new = [
    'Pre_Match_Strength', 'Opponent_Strength', 'Strength_Diff', 'home',
    'Pred_Team_Goals', 'Pred_Team_Conceded', 'Pred_Goal_Diff'
]

target_col = 'Fantavoto'

cols_to_scale = player_cols_new + team_past_cols_new + team_future_cols_new


def generate_all_sequences(all_scaled_dfs, n_past=N_PAST, m_future=M_FUTURE):
    """
    Wrapper function that iterates over all season DataFrames and generates sequences for each.
    This uses the augmented function.
    """
    all_sequences = []
    print(f"Inizio generazione sequenze (con AUGMENTATION) per {len(all_scaled_dfs)} stagioni...")

    for i, df in enumerate(all_scaled_dfs):
        # USE THE AUGMENTED FUNCTION
        X_p, X_f, y = create_sequences_dual_augmented(df, n_past, m_future)
        
        all_sequences.append((X_p, X_f, y))
        print(f"Stagione {i}: generati {len(y)} campioni.")

    return all_sequences


def prepare_sequences(scaler_type=SCALER_TYPE, n_past=N_PAST, m_future=M_FUTURE):
    """
    Master function to prepare Train, Validation, and Test sets ready for the Neural Network.
    """
    # Scale all data and get the denormalization factor
    tuple1 = prepare_all_data(scaler_type)
    scaled_dfs = tuple1[0]
    range_voti = tuple1[1]

    # Generate sliding window sequences
    sequences_list = generate_all_sequences(scaled_dfs, n_past=n_past, m_future=m_future)

    # Split into Train, Validation, and Test based on seasons 
    # 7 seasons in total: 5 for training, 1 for validation and 1 for testing
    
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
    """
    Builds the Best model for the LSTM by manually inserting the hyperparameters
    obtained during the tuning.
    """
    # Input
    input_past = Input(shape=shape_past, name='Input_Storico_LSTM')
    input_future = Input(shape=shape_future, name='Input_Contesto_Futuro')
    
    # LSTM Part 
    x = LSTM(32, return_sequences=False, activation='tanh')(input_past)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x) 
    
    # Dense part
    y = Dense(48, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y) 
    
    combined = Concatenate()([x, y])
    
    # Final Layer
    z = Dense(32, activation='relu')(combined)
    z = Dropout(0.3)(z) 
    
    # Extra Layer
    z = Dense(32, activation='relu')(z) 
    z = Dropout(0.3)(z) 
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    final_loss = get_ranking_loss(weight_mse=1.0, weight_cos=8.0)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mae', 
                  metrics=['mae', 'mse'])
    
    return model


def train_and_test_best_config(plot_graphs=False, analyze_errors=False):
    """
    Trains and tests the best possible model for the LSTM and prints the metrics.
    if necessary, it calls reliability matrix and plot graphics
    """
    print("START TRAINING OF THE BEST MODEL")

    # Prepare the data
    lista = prepare_sequences(SCALER_TYPE, N_PAST) 
    denorm_factor = lista[3]
    
    # Unpack
    train = lista[0]
    X_train_past, X_train_fut, y_train = train[0], train[1], train[2]
    
    val = lista[1]
    X_val_past, X_val_fut, y_val = val[0], val[1], val[2]
    
    test = lista[2]
    X_test_past, X_test_fut, y_test = test[0], test[1], test[2]

    shape_p = (X_train_past.shape[1], X_train_past.shape[2])
    shape_f = (X_train_fut.shape[1],)
    
    # Train the best model
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
        batch_size=128,   
        callbacks=callbacks,
        verbose=1 
    )

    print("TRAINING DONE.")

    # Plot
    if plot_graphs:
        plot_diagnostics(history, model, X_test_past, X_test_fut, y_test, denorm_factor)

    print("\n--- VALUTAZIONE FINALE ---")
    preds = model.predict([X_test_past, X_test_fut], verbose=0).flatten()
    
    mae_real = mean_absolute_error(y_test, preds) * denorm_factor
    spearman_score, _ = spearmanr(y_test, preds)

    # Print the metrics    
    print(f"MAE Reale: {mae_real:.4f}")
    print(f"Spearman:  {spearman_score:.4f}")
    
    return model, mae_real


def get_ranking_loss(weight_mse=1.0, weight_cos=5.0):
    """
    Computes the custom loss by doing a weighted multiplication of MSE and cosine similarity
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
            
            # AUGMENTATION LOGIC
            
            # Add the base case
            X_past_list.append(window_past)
            X_fut_list.append(window_future)
            y_list.append(target_val)
            
            # Over-sampling for high grades: if the grade is high we put it three more times in the sequence
            if target_val >= 10.0:
                for _ in range(3): 
                    X_past_list.append(window_past)
                    X_fut_list.append(window_future)
                    y_list.append(target_val)
                    
            # If the grade is good we add it another time
            elif target_val >= 8.0:
                X_past_list.append(window_past)
                X_fut_list.append(window_future)
                y_list.append(target_val)

    return np.array(X_past_list), np.array(X_fut_list), np.array(y_list)


if __name__ == "__main__":
    model, final_mae = train_and_test_best_config(plot_graphs=True)