import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from lstm_with_predictions import plot_diagnostics, prepare_sequences, SCALER_TYPE, N_PAST


def analyze_model_errors(y_true_scaled, y_pred_scaled, denorm_factor, lower_bound_est=4.0):
    """
    Detailed analysis of the errors of the model.
    1) Top 15 errors
    2) Errors by grades
    3) Biases (tends to overestimate or to underestimate?)
    """
    print("\n===========================================")
    print("ERROR ANALYSIS\n")
    
    # 1. Denormalization
    y_real = y_true_scaled * denorm_factor + lower_bound_est
    p_real = y_pred_scaled * denorm_factor + lower_bound_est
    
    # 2. Compute differences
    errors = y_real - p_real
    abs_errors = np.abs(errors)
    
    df_err = pd.DataFrame({
        'Reale': y_real,
        'Predetto': p_real,
        'Errore': errors,          
        'Errore_Abs': abs_errors  
    })
    
    # Top 15 worst errors
    print("\n--- TOP 15 ERRORI PIÙ GRAVI ---")
    print(df_err.sort_values('Errore_Abs', ascending=False).head(15).to_string(index=False))
    
    # Errors by grades (to understand where the model suffers the most)
    bins = [0, 5.5, 6.0, 6.5, 7.0, 8.0, 20.0]
    labels = ['< 5.5 (Malus)', '5.5-6.0 (Suff)', '6.0-6.5 (Buono)', '6.5-7.0 (Ottimo)', '7.0-8.0 (Super)', '> 8.0 (Exploit)']
    
    df_err['Fascia'] = pd.cut(df_err['Reale'], bins=bins, labels=labels)
    mae_by_bin = df_err.groupby('Fascia', observed=False)['Errore_Abs'].mean()
    
    plt.figure(figsize=(12, 5))
    sns.barplot(x=mae_by_bin.index, y=mae_by_bin.values, palette="viridis")
    plt.title('Errore Medio (MAE) per Fascia di Voto Reale')
    plt.ylabel('Errore Medio (Punti)')
    plt.xlabel('Voto Reale')
    plt.grid(axis='y', alpha=0.3)
    
    plt.show()
    
    print("\nMAE per Fascia:")
    print(mae_by_bin)

    # Biases analysis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_err['Reale'], y=df_err['Errore'], alpha=0.4)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title('Analisi dei Residui (Bias)')
    plt.xlabel('Voto Reale')
    plt.ylabel('Errore (Reale - Predetto)')
    plt.text(x=df_err['Reale'].min(), y=df_err['Errore'].max(), s="Sopra 0: Sottostima\nSotto 0: Sovrastima", bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True, alpha=0.3)
    
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
    # Input
    input_past = Input(shape=shape_past, name='Input_Storico_LSTM')
    input_future = Input(shape=shape_future, name='Input_Contesto_Futuro')
    
    # LSTM Part 
    x = LSTM(96, return_sequences=False, activation='tanh')(input_past)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x) 
    
    # Dense part
    y = Dense(16, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y) 
    
    combined = Concatenate()([x, y])
    
    # Final Layer
    z = Dense(96, activation='relu')(combined)
    z = Dropout(0.3)(z) 
    
    # Extra Layer
    z = Dense(96, activation='relu')(z) 
    z = Dropout(0.3)(z) 
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    final_loss = get_ranking_loss(weight_mse=1.0, weight_cos=10.0)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss=final_loss, 
                  metrics=['mae', 'mse'])
    
    return model


def train_and_test_best_config(plot_graphs=False):
    """
    Trains and tests the best possible model for the LSTM and calls analyze errors
    """
    print("START TRAINING OF THE BEST MODEL (With Error analysis)")
    
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

    print("TRAINING DONE.")

    if plot_graphs:
        plot_diagnostics(history, model, X_test_past, X_test_fut, y_test, denorm_factor)

    print("\nFINAL EVALUATION")
    preds = model.predict([X_test_past, X_test_fut], verbose=0).flatten()
    
    mae_real = mean_absolute_error(y_test, preds) * denorm_factor
    spearman_score, _ = spearmanr(y_test, preds)
    
    print(f"MAE: {mae_real:.4f}")
    print(f"Spearman:  {spearman_score:.4f}")    

    analyze_model_errors(y_test, preds, denorm_factor, lower_bound_est=4.0)
    
    return model, mae_real

if __name__ == "__main__":
    model, final_mae = train_and_test_best_config(plot_graphs=True)
