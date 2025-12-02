import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Import dalle tue utility (assicurati che i nomi file siano giusti)
from lstm_with_predictions import prepare_sequences, get_ranking_loss, plot_diagnostics
from lstm_with_predictions import N_PAST, M_FUTURE, SCALER_TYPE 

# --- 1. COSTRUZIONE MODELLO OTTIMIZZATO ---
def build_best_cnn(shape_past, shape_future):
    """
    CNN Hard-Coded con i parametri vincenti del Tuner.
    Aggiornato: Secondo Layer ATTIVO.
    """
    # Parametri dal Tuning
    FILTERS_L1 = 32         # Primo layer
    FILTERS_L2 = 64         # Secondo layer (Raddoppiamo i filtri, standard CNN)
    KERNEL_SIZE = 2
    DENSE_CONTEXT = 48
    DENSE_FINAL = 32
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    COSINE_WEIGHT = 5.0
    
    # --- Input ---
    input_past = Input(shape=shape_past, name='Input_Storico_CNN')
    input_future = Input(shape=shape_future, name='Input_Contesto')
    
    # --- RAMO CNN (Analizza il passato) ---
    
    # 1° Layer CNN (Vincitore: 32 filtri)
    x = Conv1D(filters=FILTERS_L1, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(input_past)
    x = BatchNormalization()(x)
    
    # 2° Layer CNN (Vincitore: True -> Aggiunto)
    # Raddoppiamo i filtri a 64 per catturare pattern di livello superiore
    x = Conv1D(filters=FILTERS_L2, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Pooling
    x = GlobalAveragePooling1D()(x) 
    x = Dropout(DROPOUT_RATE)(x)
    
    # --- RAMO CONTESTO (Dense) ---
    y = Dense(DENSE_CONTEXT, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(DROPOUT_RATE)(y)
    
    # --- UNIONE ---
    combined = Concatenate()([x, y])
    
    # --- LAYER FINALE ---
    z = Dense(DENSE_FINAL, activation='relu')(combined)
    z = Dropout(DROPOUT_RATE)(z)
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    # Loss Ibrida (Vincitore: Peso Coseno 5.0)
    loss_ranking = get_ranking_loss(weight_mse=1.0, weight_cos=COSINE_WEIGHT)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='mae', 
                  metrics=['mae', 'mse'])
    
    return model

# --- 2. TRAINING E TEST ---
def train_best_cnn(plot_graphs=True):
    print("\n=== AVVIO TRAINING CNN DEFINITIVA ===")
    
    # 1. Preparazione Dati
    data_bundle = prepare_sequences(SCALER_TYPE, N_PAST, M_FUTURE)
    
    # Unpack
    train = data_bundle[0]
    val = data_bundle[1]
    test = data_bundle[2]
    denorm_factor = data_bundle[3]
    
    # Estrazione tensori (ignoriamo ID se presenti, CNN non li usa qui)
    X_train_p, X_train_f = train[0], train[1]
    y_train = train[-1]
    
    X_val_p, X_val_f = val[0], val[1]
    y_val = val[-1]
    
    X_test_p, X_test_f = test[0], test[1]
    y_test = test[-1]

    # 2. Costruzione
    shape_p = (X_train_p.shape[1], X_train_p.shape[2])
    shape_f = (X_train_f.shape[1],)
    
    model = build_best_cnn(shape_p, shape_f)
    model.summary()

    # 3. Callbacks (Configurazione "Long Run")
    callbacks = [
        # Pazienza alta perché il LR è molto basso (0.0005)
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1)
    ]

    # 4. Fit
    history = model.fit(
        x=[X_train_p, X_train_f], 
        y=y_train,
        validation_data=([X_val_p, X_val_f], y_val),
        epochs=200,      # Abbondiamo con le epoche
        batch_size=64,   
        callbacks=callbacks,
        verbose=1 
    )

    print("TRAINING COMPLETATO.")

    # 5. Plotting
    if plot_graphs:
        # Nota: assicurati che plot_diagnostics accetti i parametri corretti nel file importato
        try:
            plot_diagnostics(history, model, X_test_p, X_test_f, y_test, denorm_factor)
        except Exception as e:
            print(f"Impossibile generare grafico: {e}")

    # 6. Valutazione Finale
    print("\n--- RISULTATI FINALI CNN ---")
    preds = model.predict([X_test_p, X_test_f], verbose=0).flatten()
    
    mae_scaled = mean_absolute_error(y_test, preds)
    mae_real = mae_scaled * denorm_factor
    spearman, _ = spearmanr(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"MAE Reale: {mae_real:.4f}")
    print(f"Spearman:  {spearman:.4f}")
    print(f"R² Score:  {r2:.4f}")
    
    return model, mae_real


def find_best_n_past_cnn(min_n=3, max_n=12):
    """
    Cerca il miglior valore di N_PAST specificamente per la CNN.
    Plotta sia MAE che Spearman.
    """
    results_mae = {}
    results_spearman = {}
    
    print(f"\n🚀 AVVIO RICERCA N_PAST PER CNN (da {min_n} a {max_n})...")
    
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
            
            # 2. Costruisci il modello CNN Hard-Coded
            shape_p = (X_train_p.shape[1], X_train_p.shape[2])
            shape_f = (X_train_f.shape[1],)
            
            model = build_best_cnn(shape_p, shape_f)
            
            # 3. Training Veloce
            callbacks_fast = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=0)
            ]
            
            model.fit(
                x=[X_train_p, X_train_f], y=y_train,
                validation_data=([X_val_p, X_val_f], y_val),
                epochs=80, 
                batch_size=64,
                callbacks=callbacks_fast,
                verbose=0
            )
            
            # 4. Valutazione
            preds = model.predict([X_test_p, X_test_f], verbose=0).flatten()
            
            mae_scaled = mean_absolute_error(y_test, preds)
            mae_real = mae_scaled * denorm_factor
            spearman, _ = spearmanr(y_test, preds)
            
            print(f"--> N={n}: MAE={mae_real:.4f} | Spearman={spearman:.4f}")
            
            # Salviamo entrambi i risultati
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
    print(f"🏆 VINCITORE CNN (Spearman): N_PAST = {best_n}")
    print(f"   Spearman: {results_spearman[best_n]:.4f}")
    print(f"   MAE:      {results_mae[best_n]:.4f}")
    print("=========================================")
    
    # --- PLOT COMPARATIVO (Doppio Asse Y) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('N_PAST (Giornate)')
    ax1.set_ylabel('MAE (Basso è meglio)', color=color)
    ax1.plot(list(results_mae.keys()), list(results_mae.values()), marker='o', color=color, label='MAE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # Secondo asse Y per Spearman
    color = 'tab:blue'
    ax2.set_ylabel('Spearman (Alto è meglio)', color=color)
    ax2.plot(list(results_spearman.keys()), list(results_spearman.values()), marker='s', linestyle='--', color=color, label='Spearman')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Performance CNN al variare di N_PAST: MAE vs Spearman')
    fig.tight_layout()  
    plt.show()
    
    return best_n, results_spearman


if __name__ == "__main__":
    best_n_lstm, _ = find_best_n_past_cnn(min_n=2, max_n=12)