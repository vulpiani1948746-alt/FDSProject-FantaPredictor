import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Import necessario per la heatmap
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from lstm_with_predictions import plot_diagnostics, prepare_sequences, SCALER_TYPE, N_PAST, M_FUTURE


def reliability_matrix(model, X_past, X_fut, y_true, denorm_factor, lower_bound=4.0):
    """
    Creates a heatmap that shows which is the probability that a grade is correct based on the grade itself
    """
    print("\nRELIABILITY MATRIX GENERATION...")
    
    # Predictions
    preds = model.predict([X_past, X_fut], verbose=0).flatten()
    
    # Denormalization
    y_real = y_true * denorm_factor + lower_bound
    p_real = preds * denorm_factor + lower_bound
    
    df = pd.DataFrame({'Reale': y_real, 'Predetto': p_real})
    
    # Binning to group grades
    df['Bin_Predetto'] = np.round(df['Predetto'] * 2) / 2
    df['Bin_Reale'] = np.round(df['Reale'] * 2) / 2
    
    # Filter extreme outliers
    df_filtered = df[(df['Bin_Predetto'] >= 5.0) & (df['Bin_Predetto'] <= 9.0)].copy()
    
    if df_filtered.empty:
        print("Attenzione: Nessun dato nel range 5.0-9.0 per la matrice.")
        return

    # Compute confusion matrix
    confusion_matrix = pd.crosstab(df_filtered['Bin_Predetto'], df_filtered['Bin_Reale'], normalize='index')
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt=".0%", cmap="YlGnBu", cbar=False)
    plt.title("Matrice di Affidabilità: Se prevedo X, cosa succede davvero?")
    plt.ylabel("Voto PREVISTO dal Modello")
    plt.xlabel("Voto REALE")
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


def build_best_model_lstm(shape_past, shape_future):
    """
    Builds the Best model for the LSTM by manually inserting the hyperparameters
    obtained during the tuning.
    """

    LSTM_LAYER = 96
    DENSE = 16
    FINAL = 96
    EXTRA = 96
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    WEIGHT = 10.0

    # Input
    input_past = Input(shape=shape_past, name='Input_Storico_LSTM')
    input_future = Input(shape=shape_future, name='Input_Contesto_Futuro')
    
    # LSTM Part 
    x = LSTM(LSTM_LAYER, return_sequences=False, activation='tanh')(input_past)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT)(x) 
    
    # Dense part
    y = Dense(DENSE, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(DROPOUT)(y) 
    
    combined = Concatenate()([x, y])
    
    # Final Layer
    z = Dense(FINAL, activation='relu')(combined)
    z = Dropout(DROPOUT)(z) 
    
    # Extra Layer
    z = Dense(EXTRA, activation='relu')(z) 
    z = Dropout(DROPOUT)(z) 
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    final_loss = get_ranking_loss(weight_mse=1.0, weight_cos=WEIGHT)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss=final_loss, 
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
    
    # Plot the matrix
    if analyze_errors:
        reliability_matrix(model, X_test_past, X_test_fut, y_test, denorm_factor)
    
    return model, mae_real

def find_best_n_past_lstm(min_n=2, max_n=12):
    """
    Iterates through the values of N_PAST to find the best one
    Plots both MAE and Spearman.
    """
    results_mae = {}
    results_spearman = {}
    
    print(f"\nSTART RESEARCH FOR THE BEST N_PAST (from {min_n} to {max_n})...")
    
    for n in range(min_n, max_n + 1):
        print(f"\n>>> Testing N_PAST = {n}...")
        
        try:
            # Generates data with new N_PAST
            data_bundle = prepare_sequences(SCALER_TYPE, n_past=n, m_future=M_FUTURE)
            
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
            
            # Builds the best model 
            shape_p = (X_train_p.shape[1], X_train_p.shape[2])
            shape_f = (X_train_f.shape[1],)
            
            model = build_best_model_lstm(shape_p, shape_f)
            
            # Training
            callbacks_fast = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=0)
            ]
            
            model.fit(
                x=[X_train_p, X_train_f], y=y_train,
                validation_data=([X_val_p, X_val_f], y_val),
                epochs=80,   
                batch_size=128, 
                callbacks=callbacks_fast,
                verbose=0      
            )
            
            # Evaluation
            preds = model.predict([X_test_p, X_test_f], verbose=0).flatten()
            
            mae_scaled = mean_absolute_error(y_test, preds)
            mae_real = mae_scaled * denorm_factor
            spearman, _ = spearmanr(y_test, preds)
            
            print(f"--> N={n}: MAE={mae_real:.4f} | Spearman={spearman:.4f}")
            
            results_mae[n] = mae_real
            results_spearman[n] = spearman
            
            tf.keras.backend.clear_session()
            del model, X_train_p
            
        except Exception as e:
            print(f"Errore con N={n}: {e}")

    # Final Result
    best_n = max(results_spearman, key=results_spearman.get)
    print("\n=========================================")
    print(f"🏆 WINNER LSTM (Spearman): N_PAST = {best_n}")
    print(f"   Spearman: {results_spearman[best_n]:.4f}")
    print(f"   MAE:      {results_mae[best_n]:.4f}")
    print("=========================================")
    
    # Plot to Compare
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('N_PAST')
    ax1.set_ylabel('MAE', color=color)
    ax1.plot(list(results_mae.keys()), list(results_mae.values()), marker='o', color=color, label='MAE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx() 
    color = 'tab:purple'
    ax2.set_ylabel('Spearman', color=color)
    ax2.plot(list(results_spearman.keys()), list(results_spearman.values()), marker='s', linestyle='--', color=color, label='Spearman')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Performance of LSTM for varying N_PAST')
    fig.tight_layout()  
    plt.show()
    
    return best_n, results_spearman


if __name__ == "__main__":
    best_n, _ = find_best_n_past_lstm(min_n=2, max_n=8)
    
    # N_PAST = best_n 
    # model, final_mae = train_and_test_best_config(plot_graphs=True)