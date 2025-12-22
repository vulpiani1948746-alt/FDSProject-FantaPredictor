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
pd.set_option('display.max_rows', None)


FOLDER = "C:/Users/franc/OneDrive/Desktop/Github/FDSProject-FantaPredictor/final_data"  # Put your folder that will contain all the data
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2021-2022", "2022-2023", "2023-2024", "2024-2025"] 

N_PAST = 4
M_FUTURE = 1
SCALER_TYPE = "minmax"
ROLE = "player"

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

player_cols_new = [
    'Voto_Fantacalcio', 'Player_Goals', 'Ast', 'Shots', 'SoT', 
    'Player_xG', 'Min', 'Touches', 'Voto_Diff', 'Age', 'Value', 'Min', 'Autogoal', 'Missed_Pen',
    'PK', 'CardY', 'CardR', 'Tkl', 'Int', 'Blocks', 'Player_xAST', 'SCA', 'GCA', 'Cmp%_x', 'PrgP',
    'PrgC', 'Cross', 'Succ', 'Fantavoto', 'Player_xG_Overperf', 'Shot_Accuracy', 'Role_A', 'Role_C',
    'Role_D', 'Voto_Diff', 'Avg_Voto_Fantacalcio', 'Avg_Fantavoto', 'Avg_Voto_Diff'
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
        df_players = pd.read_pickle(os.path.join(FOLDER, f"season{season}", f"FinalDF_{ROLE}.pkl"))
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
    

def build_model(shape_past, shape_future):
    """
    Constructs a Dual-Input Neural Network using the Keras Functional API.
    """
    # Branch 1: Historical Data (LSTM)
    input_past = Input(shape=shape_past, name='Input_Storico_LSTM')
    
    # LSTM layer to capture temporal trends
    x = LSTM(64, return_sequences=False, activation='tanh')(input_past)
    x = BatchNormalization()(x) # Normalizes activations for stable training
    x = Dropout(0.1)(x)         # Prevents overfitting
    
    # Branch 2: Future Context (Dense)
    input_future = Input(shape=shape_future, name='Input_Contesto_Futuro')
    
    # Dense layer to process static features (e.g., Opponent Strength)
    y = Dense(32, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(0.1)(y)
    
    # Combine the insights from history and future context
    combined = Concatenate()([x, y])
    
    # Fully connected layers to map features to the final vote
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.1)(z)
    z = Dense(32, activation='relu')(z)
    
    # Output layer: Single neuron with linear activation for regression (predicting a continuous value)
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    custom_loss = get_ranking_loss(weight_mse=1.0, weight_cos=5.0)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae', 'mse'])
    
    return model


def train_and_test(n_past=N_PAST, scaler_type=SCALER_TYPE, plot_graphs=False):
    """
    Main orchestration function to train and evaluate the model.
    """
    print("START TRAINING -----")
    
    # Data Preparation
    lista = prepare_sequences(scaler_type, n_past) 
    denorm_factor = lista[3]
    
    # Unpack Training Data
    train = lista[0]
    X_train_past, X_train_fut, y_train = train[0], train[1], train[2]
    
    # Unpack Validation Data
    val = lista[1]
    X_val_past, X_val_fut, y_val = val[0], val[1], val[2]
    
    # Unpack Test Data
    test = lista[2]
    X_test_past, X_test_fut, y_test = test[0], test[1], test[2]

    # baseline_regression(X_train_past, X_train_fut, y_train, X_test_past, X_test_fut, y_test, denorm_factor)

    # Build Model
    shape_p = (X_train_past.shape[1], X_train_past.shape[2]) # (Timesteps, Features)
    shape_f = (X_train_fut.shape[1],)                       # (Context Features)
    model = build_model(shape_p, shape_f)

    # Training Configuration
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

    # Plotting
    if plot_graphs:
        plot_diagnostics(history, model, X_test_past, X_test_fut, y_test, denorm_factor)

    # Final Evaluation
    lstm_preds = model.predict([X_test_past, X_test_fut], verbose=0).flatten()
    
    lstm_mae = mean_absolute_error(y_test, lstm_preds) * denorm_factor
    lstm_spearman, _ = spearmanr(y_test, lstm_preds)
    
    print(f"\n--- LSTM RESULTS ---")
    print(f"Real MAE:   {lstm_mae:.4f}")
    print(f"Spearman:   {lstm_spearman:.4f}")
    
    return model, lstm_mae


def plot_diagnostics(history, model, X_past, X_fut, y_true, denorm_factor, lower_bound=0):
    """
    Visualizes training performance and prediction accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Learning Curves
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', linewidth=2)
    ax1.set_title('Learning Curve (Loss)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MAE (Absolute Error)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sample Predictions
    n_samples = 20
    if len(y_true) < n_samples: n_samples = len(y_true)
    
    # Select random indices for sampling
    indices = np.random.choice(len(y_true), n_samples, replace=False)
    
    # Extract sample data
    X_p_sample = X_past[indices]
    X_f_sample = X_fut[indices]
    y_true_sample = y_true[indices]
    
    # Generate predictions for sample
    preds_sample = model.predict([X_p_sample, X_f_sample], verbose=0).flatten()
    
    # Denormalize to get real vote values (e.g., convert 0.5 back to 6.5)
    y_real = y_true_sample * denorm_factor + lower_bound
    preds_real = preds_sample * denorm_factor + lower_bound
    
    x_axis = np.arange(n_samples)
     
    ax2.plot(x_axis, y_real, marker='o', linestyle='-', color='grey', alpha=0.5, label='Real Vote')
    ax2.scatter(x_axis, preds_real, color='red', s=100, marker='x', label='AI Prediction', zorder=10)
    ax2.vlines(x_axis, ymin=y_real, ymax=preds_real, colors='orange', alpha=0.4)
    
    ax2.set_title(f'Reality vs Prediction (Sample of {n_samples})')
    ax2.set_ylabel('Fantacalcio Vote')
    ax2.set_xticks(x_axis)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


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

if __name__ == "__main__":
    train_and_test(plot_graphs=True)