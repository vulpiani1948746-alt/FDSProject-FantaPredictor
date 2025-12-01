import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

# Import dalle tue utility (assicurati che i nomi file siano giusti)
from lstm_with_predictions import prepare_sequences, get_ranking_loss, plot_diagnostics
from lstm_with_predictions import N_PAST, M_FUTURE, SCALER_TYPE 

# --- 1. COSTRUZIONE MODELLO OTTIMIZZATO ---
def build_best_cnn(shape_past, shape_future):
    """
    CNN Hard-Coded con i parametri vincenti del Tuner.
    """
    # Parametri dal Tuning
    FILTERS = 96
    KERNEL_SIZE = 3
    DENSE_CONTEXT = 48
    DENSE_FINAL = 64
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.0005
    COSINE_WEIGHT = 10.0
    
    # --- Input ---
    input_past = Input(shape=shape_past, name='Input_Storico_CNN')
    input_future = Input(shape=shape_future, name='Input_Contesto')
    
    # --- RAMO CNN (Analizza il passato) ---
    # Layer 1 (Vincitore: 96 filtri, Kernel 3)
    x = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(input_past)
    x = BatchNormalization()(x)
    
    # Second Layer? (Vincitore: False -> Saltiamo il secondo blocco Conv1D)
    
    # Pooling
    x = GlobalAveragePooling1D()(x) 
    x = Dropout(DROPOUT_RATE)(x)
    
    # --- RAMO CONTESTO (Dense) ---
    # Vincitore: 48 neuroni
    y = Dense(DENSE_CONTEXT, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(DROPOUT_RATE)(y)
    
    # --- UNIONE ---
    combined = Concatenate()([x, y])
    
    # --- LAYER FINALE ---
    # Vincitore: 64 neuroni
    z = Dense(DENSE_FINAL, activation='relu')(combined)
    z = Dropout(DROPOUT_RATE)(z)
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    # Loss Ibrida (Vincitore: Peso Coseno 10.0)
    loss_ranking = get_ranking_loss(weight_mse=1.0, weight_cos=COSINE_WEIGHT)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss=loss_ranking, 
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

if __name__ == "__main__":
    final_model, mae = train_best_cnn(plot_graphs=True)