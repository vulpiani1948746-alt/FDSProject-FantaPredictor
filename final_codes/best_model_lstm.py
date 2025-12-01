from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from lstm_with_predictions import plot_diagnostics, prepare_sequences, SCALER_TYPE, N_PAST

def get_ranking_loss(weight_mse=1.0, weight_cos=5.0):
    mse_fn = tf.keras.losses.MeanSquaredError()
    cosine_fn = tf.keras.losses.CosineSimilarity()

    def ranking_loss(y_true, y_pred):
        loss_mse = mse_fn(y_true, y_pred)
        loss_cos = cosine_fn(y_true, y_pred)
        return (weight_mse * loss_mse) + (weight_cos * loss_cos)

    return ranking_loss

def build_best_model(shape_past, shape_future):
    # --- 1. INPUT ---
    input_past = Input(shape=shape_past, name='Input_Storico_LSTM')
    input_future = Input(shape=shape_future, name='Input_Contesto_Futuro')
    
    # --- 2. RAMO LSTM (Vincitore: 64 units, Dropout 0.4) ---
    x = LSTM(96, return_sequences=False, activation='tanh')(input_past)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x) # <-- Dropout alto qui
    
    # --- 3. RAMO CONTESTO (Vincitore: 16 units, Dropout 0.4) ---
    y = Dense(16, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y) # <-- E qui
    
    # --- 4. UNIONE ---
    combined = Concatenate()([x, y])
    
    # --- 5. LAYER FINALE 1 (Vincitore: 128 units, Dropout 0.4) ---
    z = Dense(96, activation='relu')(combined)
    z = Dropout(0.3)(z) # <-- E qui
    
    # --- 6. LAYER EXTRA (Vincitore: True) ---
    # Il tuner aggiungeva un layer grande la metà del precedente
    z = Dense(96, activation='relu')(z) # 128 // 2 = 64
    z = Dropout(0.3)(z) # <-- E anche qui
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    # --- 7. COMPILAZIONE (Vincitori: LR 0.001, Cosine Wgt 5.0) ---
    final_loss = get_ranking_loss(weight_mse=1.0, weight_cos=10.0)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss=final_loss, 
                  metrics=['mae', 'mse'])
    
    return model


def train_and_test_best_config(plot_graphs=False):
    print("--- AVVIO TRAINING MODELLO OTTIMIZZATO (NO EMBEDDING) ---")
    
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
    
    model = build_best_model(shape_p, shape_f)

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
    
    return model, mae_real


if __name__ == "__main__":
    model, final_mae = train_and_test_best_config(plot_graphs=True)