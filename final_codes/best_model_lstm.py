import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Import necessario per la heatmap
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from lstm_with_predictions import plot_diagnostics, prepare_sequences, SCALER_TYPE, N_PAST, M_FUTURE

# --- NUOVA FUNZIONE MATRICE DI AFFIDABILITÀ ---
def reliability_matrix(model, X_past, X_fut, y_true, denorm_factor, lower_bound=4.0):
    """
    Crea una heatmap che mostra la probabilità reale di un voto
    dato il voto previsto dal modello.
    """
    print("\n--- GENERAZIONE MATRICE DI AFFIDABILITÀ ---")
    
    # 1. Predizioni
    preds = model.predict([X_past, X_fut], verbose=0).flatten()
    
    # 2. Denormalizzazione
    y_real = y_true * denorm_factor + lower_bound
    p_real = preds * denorm_factor + lower_bound
    
    # 3. Creazione DataFrame
    df = pd.DataFrame({'Reale': y_real, 'Predetto': p_real})
    
    # 4. Binning (Arrotondiamo i voti per raggrupparli)
    # Arrotondiamo al mezzo punto più vicino (6.0, 6.5, 7.0...)
    df['Bin_Predetto'] = np.round(df['Predetto'] * 2) / 2
    df['Bin_Reale'] = np.round(df['Reale'] * 2) / 2
    
    # Filtriamo outlier estremi per pulizia grafico (visualizziamo tra 5.0 e 10.0)
    # Puoi allargare il range se vuoi vedere anche i 4 o i 13
    df_filtered = df[(df['Bin_Predetto'] >= 5.0) & (df['Bin_Predetto'] <= 9.0)].copy()
    
    if df_filtered.empty:
        print("Attenzione: Nessun dato nel range 5.0-9.0 per la matrice.")
        return

    # 5. Calcolo della Matrice di Confusione (Normalizzata per riga)
    # "Quando prevedo X (righe), quante volte accade Y (colonne)?"
    confusion_matrix = pd.crosstab(df_filtered['Bin_Predetto'], df_filtered['Bin_Reale'], normalize='index')
    
    # 6. Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt=".0%", cmap="YlGnBu", cbar=False)
    plt.title("Matrice di Affidabilità: Se prevedo X, cosa succede davvero?")
    plt.ylabel("Voto PREVISTO dal Modello")
    plt.xlabel("Voto REALE")
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
                  loss='mae', 
                  metrics=['mae', 'mse'])
    
    return model

# --- TRAINING & TEST ---
def train_and_test_best_config(plot_graphs=False, analyze_errors=False):
    print("--- AVVIO TRAINING MODELLO OTTIMIZZATO ---")
    
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
        batch_size=128,   
        callbacks=callbacks,
        verbose=1 
    )

    print("TRAINING COMPLETATO.")

    # --- 1. PLOT STANDARD (Curve e Scatter) ---
    if plot_graphs:
        plot_diagnostics(history, model, X_test_past, X_test_fut, y_test, denorm_factor)

    print("\n--- VALUTAZIONE FINALE ---")
    preds = model.predict([X_test_past, X_test_fut], verbose=0).flatten()
    
    mae_real = mean_absolute_error(y_test, preds) * denorm_factor
    spearman_score, _ = spearmanr(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"MAE Reale: {mae_real:.4f}")
    print(f"Spearman:  {spearman_score:.4f}")
    
    # --- 2. ANALISI PROFONDA ERRORI ---
    if analyze_errors:
        # Usiamo 3.5 come stima del minimo (o estrailo dallo scaler se preferisci)
        reliability_matrix(model, X_test_past, X_test_fut, y_test, denorm_factor)
    
    return model, mae_real

def find_best_n_past_lstm(min_n=2, max_n=12):
    """
    Cerca il miglior valore di N_PAST per il modello LSTM ottimizzato.
    """
    results_mae = {}
    results_spearman = {}
    
    print(f"\n🚀 AVVIO RICERCA N_PAST PER LSTM (da {min_n} a {max_n})...")
    
    for n in range(min_n, max_n + 1):
        print(f"\n>>> Testing N_PAST = {n}...")
        
        try:
            # 1. Rigenera i dati con il nuovo N
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
            
            # 2. Costruisci il modello LSTM Hard-Coded
            # Nota: Usa la tua funzione build_best_model_lstm esistente
            shape_p = (X_train_p.shape[1], X_train_p.shape[2])
            shape_f = (X_train_f.shape[1],)
            
            model = build_best_model_lstm(shape_p, shape_f)
            
            # 3. Training Veloce (Fast Track)
            callbacks_fast = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=0)
            ]
            
            model.fit(
                x=[X_train_p, X_train_f], y=y_train,
                validation_data=([X_val_p, X_val_f], y_val),
                epochs=80,       # Meno epoche del training finale (abbastanza per vedere il trend)
                batch_size=128,  # Batch grande per velocità e stabilità cosine loss
                callbacks=callbacks_fast,
                verbose=0        # Silenzioso
            )
            
            # 4. Valutazione
            preds = model.predict([X_test_p, X_test_f], verbose=0).flatten()
            
            mae_scaled = mean_absolute_error(y_test, preds)
            mae_real = mae_scaled * denorm_factor
            spearman, _ = spearmanr(y_test, preds)
            
            print(f"--> N={n}: MAE={mae_real:.4f} | Spearman={spearman:.4f}")
            
            results_mae[n] = mae_real
            results_spearman[n] = spearman
            
            # Pulizia RAM
            tf.keras.backend.clear_session()
            del model, X_train_p
            
        except Exception as e:
            print(f"Errore con N={n}: {e}")

    # --- RISULTATO FINALE (Basato su Spearman) ---
    best_n = max(results_spearman, key=results_spearman.get)
    print("\n=========================================")
    print(f"🏆 VINCITORE LSTM (Spearman): N_PAST = {best_n}")
    print(f"   Spearman: {results_spearman[best_n]:.4f}")
    print(f"   MAE:      {results_mae[best_n]:.4f}")
    print("=========================================")
    
    # --- PLOT COMPARATIVO ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('N_PAST (Giornate)')
    ax1.set_ylabel('MAE (Basso è meglio)', color=color)
    ax1.plot(list(results_mae.keys()), list(results_mae.values()), marker='o', color=color, label='MAE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # Secondo asse Y per Spearman
    color = 'tab:purple'
    ax2.set_ylabel('Spearman (Alto è meglio)', color=color)
    ax2.plot(list(results_spearman.keys()), list(results_spearman.values()), marker='s', linestyle='--', color=color, label='Spearman')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Performance LSTM al variare di N_PAST')
    fig.tight_layout()  
    plt.show()
    
    return best_n, results_spearman

# --- USO ---
if __name__ == "__main__":
    best_n, _ = find_best_n_past_lstm(min_n=2, max_n=12)
    
    # E poi usa il vincitore nel training finale:
    # N_PAST = best_n 
    # model, final_mae = train_and_test_best_config(plot_graphs=True, analyze_errors=True)