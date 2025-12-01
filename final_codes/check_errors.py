import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Assicurati di avere seaborn installato (pip install seaborn)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from lstm_with_predictions import plot_diagnostics, prepare_sequences, SCALER_TYPE, N_PAST

# --- NUOVA FUNZIONE DI ANALISI ERRORI ---
def analyze_model_errors(y_true_scaled, y_pred_scaled, denorm_factor, lower_bound_est=3.5):
    """
    Analisi dettagliata degli errori del modello.
    lower_bound_est: stima del voto minimo (es. 3.5) per rendere i grafici leggibili in scala reale.
    """
    print("\n===========================================")
    print("🔎 ANALISI APPROFONDITA ERRORI")
    print("===========================================")
    
    # 1. Denormalizzazione
    y_real = y_true_scaled * denorm_factor + lower_bound_est
    p_real = y_pred_scaled * denorm_factor + lower_bound_est
    
    # 2. Calcolo Differenze
    errors = y_real - p_real
    abs_errors = np.abs(errors)
    
    df_err = pd.DataFrame({
        'Reale': y_real,
        'Predetto': p_real,
        'Errore': errors,          # Se < 0: Sovrastima (Pred > Reale)
        'Errore_Abs': abs_errors   # Grandezza errore
    })
    
    # --- A. TOP 15 ERRORI PEGGIORI ---
    print("\n--- TOP 15 ERRORI PIÙ GRAVI ---")
    print(df_err.sort_values('Errore_Abs', ascending=False).head(15).to_string(index=False))
    
    # --- B. ERRORE PER FASCIA DI VOTO (Binning) ---
    # Creiamo fasce di voto per vedere dove il modello soffre
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

    # --- C. ANALISI RESIDUI (Bias) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_err['Reale'], y=df_err['Errore'], alpha=0.4)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title('Analisi dei Residui (Bias)')
    plt.xlabel('Voto Reale')
    plt.ylabel('Errore (Reale - Predetto)')
    plt.text(x=df_err['Reale'].min(), y=df_err['Errore'].max(), s="Sopra 0: Sottostima\nSotto 0: Sovrastima", bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True, alpha=0.3)
    
    plt.show()

# --- LOSS CUSTOM ---
def get_ranking_loss(weight_mse=1.0, weight_cos=5.0):
    mse_fn = tf.keras.losses.MeanSquaredError()
    cosine_fn = tf.keras.losses.CosineSimilarity()

    def ranking_loss(y_true, y_pred):
        loss_mse = mse_fn(y_true, y_pred)
        loss_cos = cosine_fn(y_true, y_pred)
        return (weight_mse * loss_mse) + (weight_cos * loss_cos)

    return ranking_loss

# --- COSTRUZIONE MODELLO ---
def build_best_model_lstm(shape_past, shape_future):
    # --- 1. INPUT ---
    input_past = Input(shape=shape_past, name='Input_Storico_LSTM')
    input_future = Input(shape=shape_future, name='Input_Contesto_Futuro')
    
    # --- 2. RAMO LSTM ---
    x = LSTM(96, return_sequences=False, activation='tanh')(input_past)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x) 
    
    # --- 3. RAMO CONTESTO ---
    y = Dense(16, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y) 
    
    # --- 4. UNIONE ---
    combined = Concatenate()([x, y])
    
    # --- 5. LAYER FINALE ---
    z = Dense(96, activation='relu')(combined)
    z = Dropout(0.3)(z) 
    
    # --- 6. LAYER EXTRA ---
    z = Dense(96, activation='relu')(z) 
    z = Dropout(0.3)(z) 
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    # --- 7. COMPILAZIONE ---
    final_loss = get_ranking_loss(weight_mse=1.0, weight_cos=10.0)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss=final_loss, 
                  metrics=['mae', 'mse'])
    
    return model

# --- TRAINING & TEST ---
def train_and_test_best_config(plot_graphs=False):
    print("--- AVVIO TRAINING MODELLO OTTIMIZZATO (Con Error Analysis) ---")
    
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
    r2 = r2_score(y_test, preds)
    
    print(f"MAE Reale: {mae_real:.4f}")
    print(f"Spearman:  {spearman_score:.4f}")
    print(f"R² Score:  {r2:.4f}")
    
    # --- CHIAMATA ANALISI ERRORI ---
    # Nota: lower_bound_est=3.5 è una stima tipica del voto minimo nel fantacalcio 
    # per rendere i grafici realistici (range 3.5 - 10.5 circa).
    analyze_model_errors(y_test, preds, denorm_factor, lower_bound_est=3.5)
    
    return model, mae_real

if __name__ == "__main__":
    model, final_mae = train_and_test_best_config(plot_graphs=True)
