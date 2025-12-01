import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from lstm_with_predictions import get_ranking_loss, prepare_sequences, SCALER_TYPE, N_PAST, M_FUTURE

# Assicurati di avere queste variabili globali o passale alla funzione
# (X_train_past, X_train_fut, y_train, etc.)

def build_cnn_for_tuner(hp):
    """
    Builder function per Keras Tuner specifica per la CNN.
    """
    # Dimensioni input (dalle variabili globali)
    shape_p = (X_train_past.shape[1], X_train_past.shape[2])
    shape_f = (X_train_fut.shape[1],)

    # --- RAMO CNN (Analizza il passato) ---
    input_past = Input(shape=shape_p, name='Input_Storico_CNN')
    
    # Tuning 1: Numero di Filtri (32 - 128)
    # Più filtri = più pattern complessi riconosciuti
    hp_filters = hp.Int('filters', min_value=32, max_value=128, step=32)
    
    # Tuning 2: Kernel Size (2 o 3)
    # 2 = guarda coppie di partite; 3 = guarda terne
    hp_kernel = hp.Choice('kernel_size', values=[2, 3])
    
    # Primo Layer Conv1D
    x = Conv1D(filters=hp_filters, kernel_size=hp_kernel, activation='relu', padding='same')(input_past)
    x = BatchNormalization()(x)
    
    # Tuning 3: Secondo Layer CNN (Opzionale)
    # A volte un solo layer basta per sequenze corte (n_past=6)
    if hp.Boolean('use_second_cnn_layer'):
        # Spesso si raddoppiano i filtri nel secondo layer
        x = Conv1D(filters=hp_filters * 2, kernel_size=hp_kernel, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
    
    # Pooling
    x = GlobalAveragePooling1D()(x)
    
    # Tuning 4: Dropout (0.0 - 0.4)
    hp_dropout = hp.Float('dropout_rate', min_value=0.0, max_value=0.4, step=0.1)
    x = Dropout(hp_dropout)(x)
    
    # --- RAMO CONTESTO (Dense) ---
    input_future = Input(shape=shape_f, name='Input_Contesto')
    
    # Tuning 5: Neuroni Context
    hp_dense_ctx = hp.Int('dense_ctx_units', min_value=16, max_value=64, step=16)
    y = Dense(hp_dense_ctx, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(hp_dropout)(y) # Usiamo lo stesso dropout per coerenza
    
    # --- UNIONE ---
    combined = Concatenate()([x, y])
    
    # --- LAYER FINALE ---
    # Tuning 6: Neuroni Finali
    hp_dense_final = hp.Int('dense_final_units', min_value=32, max_value=128, step=32)
    z = Dense(hp_dense_final, activation='relu')(combined)
    z = Dropout(hp_dropout)(z)
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    # Tuning 7: Learning Rate
    hp_lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
    
    # Tuning 8: Peso Cosine Loss (Ranking)
    hp_weight_cos = hp.Choice('loss_weight_cosine', values=[5.0, 8.0, 10.0, 12.0])
    
    # Usiamo la tua loss personalizzata
    # Assumiamo che get_ranking_loss sia importata/definita
    custom_loss = get_ranking_loss(weight_mse=1.0, weight_cos=hp_weight_cos)
    
    model.compile(optimizer=Adam(learning_rate=hp_lr), 
                  loss='mae', 
                  metrics=['mae', 'mse'])
    
    return model

def run_cnn_tuning():
    print("\n--- AVVIO TUNING CNN ---")
    
    # 1. Preparazione Dati (Assicurati che prepare_sequences sia importata)
    # Nota: prepare_sequences per la CNN NON deve usare l'embedding, quindi il return avrà 4 elementi
    data_bundle = prepare_sequences(SCALER_TYPE, N_PAST, M_FUTURE)
    train = data_bundle[0]
    val = data_bundle[1]
    
    # Rendiamo globali per farli vedere a build_cnn_for_tuner
    global X_train_past, X_train_fut, y_train, X_val_past, X_val_fut, y_val
    X_train_past, X_train_fut = train[0], train[1]
    y_train = train[-1]
    X_val_past, X_val_fut = val[0], val[1]
    y_val = val[-1]
    
    # 2. Configura Tuner
    tuner = kt.Hyperband(
        build_cnn_for_tuner,
        objective=kt.Objective("val_loss", direction="min"), # Ottimizziamo la Loss Ibrida
        max_epochs=40,
        factor=3,
        directory='my_tuning_dir',
        project_name='fanta_cnn_v1',
        overwrite=True
    )
    
    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    
    # 3. Cerca
    tuner.search(
        [X_train_past, X_train_fut], y_train,
        validation_data=([X_val_past, X_val_fut], y_val),
        epochs=40,
        batch_size=64,
        callbacks=[stop_early],
        verbose=1
    )
    
    # 4. Risultati
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"""
    ✅ TUNING CNN COMPLETATO! Migliori Parametri:
    ---------------------------------------------
    Filters:           {best_hps.get('filters')}
    Kernel Size:       {best_hps.get('kernel_size')}
    Second Layer?:     {best_hps.get('use_second_cnn_layer')}
    Dense Context:     {best_hps.get('dense_ctx_units')}
    Dense Final:       {best_hps.get('dense_final_units')}
    Dropout Rate:      {best_hps.get('dropout_rate')}
    Learning Rate:     {best_hps.get('learning_rate')}
    Cosine Weight:     {best_hps.get('loss_weight_cosine')}
    """)
    
    return best_hps

if __name__ == "__main__":
    best_params = run_cnn_tuning()