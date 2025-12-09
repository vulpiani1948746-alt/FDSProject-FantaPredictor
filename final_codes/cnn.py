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
    Unites the dataframes and scales data using the existing scaler.
    Returns scaled dataframe and the used scaler.
    """
    merged_df = merge_players_with_teams(player_df, team_df)
    
    merged_df = merged_df.sort_values(by=['Player_id', 'Week_id'])
    
    if scaler is not None:
        merged_df = scaler.transform(merged_df)
        
    return merged_df, scaler


def prepare_all_data(scaler_type=SCALER_TYPE):
    """
    Loads data for all seasons, fits a global scaler, and scales the data.
    Calculates the denormalization factor needed to convert predictions back to real values.
    """
    all_data = []
    
    # Create and fit the scaler on the entire dataset
    global_scaler = create_scaler(player_cols_new, team_past_cols_new, team_future_cols_new, scaler_type, SEASONS, FOLDER)

    print("Scaler created. Starting season processing...")

    # Iterate through each season to load and scale data
    for season in SEASONS:
        df_players = pd.read_pickle(os.path.join(FOLDER, f"season{season}", "FinalDF_player.pkl"))
        df_teams = pd.read_pickle(os.path.join(FOLDER, f"season{season}", "teams.pkl"))
        
        # Scale the current season's data using the global scaler
        df_final_scaled, _ = prepare_data_dual_input(
            df_players, 
            df_teams, 
            scaler=global_scaler,
            fit=False
        )

        all_data.append(df_final_scaled)
    
    #idx_target = -1
    target_col_name = 'Fantavoto'
    
    if isinstance(global_scaler, RobustMinMaxScaler):
        # Custom scaler logic: range = upper_bound - lower_bound
        lower, upper = global_scaler.bounds[target_col_name]
        denorm_factor = upper - lower
        
    elif isinstance(global_scaler, MinMaxScaler):
        # Standard MinMax: data_range_ stores (max - min)
        cols_to_scale = player_cols_new + team_past_cols_new + team_future_cols_new
        idx = cols_to_scale.index(target_col_name)
        denorm_factor = global_scaler.data_range_[idx]
        
    elif isinstance(global_scaler, StandardScaler):
        # Standard Scaler: scale_ stores the standard deviation
        cols_to_scale = player_cols_new + team_past_cols_new + team_future_cols_new
        idx = cols_to_scale.index(target_col_name)
        denorm_factor = global_scaler.scale_[idx]

    return all_data, denorm_factor
    

def create_sequences_dual(df, n_past=N_PAST, m_future=M_FUTURE):
    """
    Generates sliding window sequences for the Dual-Input model (LSTM + Dense).
    Handles missing values (NaNs) and structures the data into (Past, Future, Target).
    """
    X_past_list = []
    X_fut_list = []
    y_list = []
    
    # Threshold for dropping a sequence if too many past values are NaN (e.g., > 50%)
    threshold_nan = math.ceil(n_past/2)
    
    # Define feature groups
    past_features = player_cols_new + team_past_cols_new 
    future_features = team_future_cols_new           
    
    idx_voto = player_cols_new.index('Fantavoto')
    
    # Group by Player to ensure sequences don't mix different players
    for player_id, group in df.groupby('Player_id'):
        
        data_past = group[past_features].values
        data_future_ctx = group[future_features].values
        data_target = group[target_col].values
        
        n_rows = len(group)
        
        # Skip players with not enough history to form a single sequence
        if n_rows < n_past + m_future:
            continue
            
        # Sliding window loop
        for i in range(n_past, n_rows - m_future + 1):
            
            window_past = data_past[i-n_past : i].copy()
            
            # NaN Handling
            player_sub_window = window_past[:, :len(player_cols_new)]
            voti_window = player_sub_window[:, idx_voto]
            num_nan = np.isnan(voti_window).sum()
            
            # If too many NaNs, discard this sample (it's too noisy)
            if num_nan >= threshold_nan: continue 
            
            # If a few NaNs, fill them with the column mean of the current window
            if num_nan > 0:
                col_means = np.nanmean(player_sub_window, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                inds = np.where(np.isnan(player_sub_window))
                player_sub_window[inds] = np.take(col_means, inds[1])
                window_past[:, :len(player_cols_new)] = player_sub_window
            
            # Final cleanup for any remaining NaNs
            if np.isnan(window_past).any():
                window_past = np.nan_to_num(window_past, nan=0.0)
            
            # Prepare Future Context (Dense Input)
            window_future = data_future_ctx[i : i+m_future].flatten()

            if np.isnan(window_future).any():
                window_future = np.nan_to_num(window_future, nan=0.0)
            
            # Prepare Target
            targets_future = data_target[i : i+m_future]
            if np.isnan(targets_future).any(): continue
            
            X_past_list.append(window_past)
            X_fut_list.append(window_future)
            y_list.append(np.mean(targets_future))

    return np.array(X_past_list), np.array(X_fut_list), np.array(y_list)


def generate_all_sequences(all_scaled_dfs, n_past=N_PAST, m_future=M_FUTURE):
    """
    Wrapper function that iterates over all season DataFrames and generates sequences for each.
    """
    all_sequences = []
    
    print(f"Generating sequences for {len(all_scaled_dfs)} seasons...")

    for i, df in enumerate(all_scaled_dfs):
        X_p, X_f, y = create_sequences_dual(df, n_past, m_future)
        
        all_sequences.append((X_p, X_f, y))
        
        print(f"Season {i}: generated {len(y)} samples.")

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

def build_cnn_model(shape_past, shape_future):
    """
    Builds a 1D CNN for time series.
    """
    input_past = Input(shape=shape_past, name='Input_Storico_CNN')
    input_future = Input(shape=shape_future, name='Input_Contesto')
    
    # CNN Part (for the past)
    # Layer 1
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_past)
    x = BatchNormalization()(x)
    
    # Layer 2
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Pooling
    x = GlobalAveragePooling1D()(x) 
    x = Dropout(0.2)(x)
    
    # Dense Layer (for future)
    y = Dense(32, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)
    
    combined = Concatenate()([x, y])
    
    # Final Layer
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.1)(z)
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    loss_ranking = get_ranking_loss(weight_mse=1.0, weight_cos=5.0)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss=loss_ranking, 
                  metrics=['mae', 'mse'])
    
    return model


def train_and_test_cnn(plot_graphs=True):
    """
    Trains and tests the CNN model and prints the evaluation metrics.
    """
    print("\nSTART TRAINING CNN MODEL")
    
    # Prepare Data
    
    data_bundle = prepare_sequences(SCALER_TYPE, N_PAST, M_FUTURE)
    
    # Unpack
    train = data_bundle[0]
    val = data_bundle[1]
    test = data_bundle[2]
    denorm_factor = data_bundle[3]
    
    X_train_p, X_train_f = train[0], train[1]
    y_train = train[-1]
    
    X_val_p, X_val_f = val[0], val[1]
    y_val = val[-1]
    
    X_test_p, X_test_f = test[0], test[1]
    y_test = test[-1]

    # Build the model
    shape_p = (X_train_p.shape[1], X_train_p.shape[2])
    shape_f = (X_train_f.shape[1],)
    
    model = build_cnn_model(shape_p, shape_f)
    # model.summary() 

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-6, verbose=1)
    ]

    # Fitting
    history = model.fit(
        x=[X_train_p, X_train_f], 
        y=y_train,
        validation_data=([X_val_p, X_val_f], y_val),
        epochs=150,      
        batch_size=64,   
        callbacks=callbacks,
        verbose=1 
    )

    print("TRAINING CNN DONE.")

    # Plotting
    if plot_graphs:
        plot_diagnostics(history, model, X_test_p, X_test_f, y_test, denorm_factor)

    # Final Evaluation
    print("\nCNN RESULTS:\n")
    preds = model.predict([X_test_p, X_test_f], verbose=0).flatten()
    
    mae_scaled = mean_absolute_error(y_test, preds)
    mae_real = mae_scaled * denorm_factor
    spearman, _ = spearmanr(y_test, preds)
    
    print(f"MAE: {mae_real:.4f}")
    print(f"Spearman:  {spearman:.4f}")
    
    return model, mae_real

if __name__ == "__main__":
    model, final_mae = train_and_test_cnn(plot_graphs=True)