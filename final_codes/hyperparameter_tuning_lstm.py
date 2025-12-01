import matplotlib.pyplot as plt
from lstm_with_predictions import train_and_test, N_PAST, SCALER_TYPE, prepare_sequences, get_ranking_loss
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt


def benchmark_scalers():
    scalers_to_test = ['robust', 'minmax', 'standard']
    results = {}
    
    for s_type in scalers_to_test:
        try:
            mae = train_and_test(n_past=N_PAST, scaler_type=s_type)
            results[s_type] = mae
        except Exception as e:
            print(f"Errore con {s_type}: {e}")

    print("\n=== CLASSIFICA FINALE SCALER ===")
    for s_type, mae in sorted(results.items(), key=lambda x: x[1]):
        print(f"{s_type.upper()}: {mae:.4f}")
        
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=['blue', 'orange', 'green'])
    plt.title('Confronto Performance Scaler (MAE)')
    plt.ylabel('Errore Voto Reale')
    plt.ylim(0.35, 0.60)
    plt.grid(axis='y', alpha=0.3)
    plt.show()


def find_best_n_past(min_n=4, max_n=12):
    results = {}
    
    print(f"Avvio ricerca dell'ottimo n_past tra {min_n} e {max_n}...")
    
    for n in range(min_n, max_n + 1):
        print(f"\n\nLet's try n = {n}\n")
        try:
            mae = train_and_test(n_past=n, plot_graphs=False)
            results[n] = mae
        except Exception as e:
            print(f"Errore con n={n}: {e}")
            results[n] = float('inf')

    best_n = min(results, key=results.get)
    best_mae = results[best_n]
    
    print("\n" + "="*30)
    print(f"MIGLIOR RISULTATO: n_past = {best_n} con MAE = {best_mae:.4f}")
    print("="*30 + "\n")

    plt.figure(figsize=(10, 6))
    x_vals = list(results.keys())
    y_vals = list(results.values())
    
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
    plt.axvline(best_n, color='r', linestyle='--', label=f'Best n={best_n}')
    
    plt.title('Performance del Modello al variare di n_past')
    plt.xlabel('Numero di partite passate (n_past)')
    plt.ylabel('Errore MAE (Reale)')
    plt.xticks(x_vals)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return results

def build_model_for_tuner(hp):
    """
    Costruisce un modello Keras con iperparametri variabili per il tuning.
    hp: HyperParameters object di Keras Tuner
    """
    shape_p = (X_train_past.shape[1], X_train_past.shape[2])
    shape_f = (X_train_fut.shape[1],)

    input_past = Input(shape=shape_p, name='Input_Storico_LSTM')
    
    hp_lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
    x = LSTM(hp_lstm_units, return_sequences=False, activation='tanh')(input_past)
    x = BatchNormalization()(x)
    
    hp_dropout = hp.Float('dropout_rate', min_value=0.0, max_value=0.4, step=0.1)
    x = Dropout(hp_dropout)(x)
    
    input_future = Input(shape=shape_f, name='Input_Contesto_Futuro')
    
    hp_dense_ctx = hp.Int('dense_ctx_units', min_value=16, max_value=64, step=16)
    y = Dense(hp_dense_ctx, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(hp_dropout)(y)
    
    combined = Concatenate()([x, y])
    
    hp_dense_final = hp.Int('dense_final_units', min_value=32, max_value=128, step=32)
    z = Dense(hp_dense_final, activation='relu')(combined)
    z = Dropout(hp_dropout)(z)
    
    if hp.Boolean('use_extra_dense_layer'):
        z = Dense(hp_dense_final // 2, activation='relu')(z)
        z = Dropout(hp_dropout)(z)
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    hp_lr = hp.Choice('learning_rate', values=[1e-3, 5e-4])
    
    hp_weight_cos = hp.Choice('loss_weight_cosine', values=[3.0, 5.0, 8.0, 10.0])
    custom_loss = get_ranking_loss(weight_mse=1.0, weight_cos=hp_weight_cos)
    
    model.compile(optimizer=Adam(learning_rate=hp_lr), 
                  loss=custom_loss, 
                  metrics=['mae', 'mse'])
    
    return model

def run_hyperparameter_tuning():
    print("\n--- AVVIO HYPERPARAMETER TUNING ---")
    
    global X_train_past, X_train_fut, y_train, X_val_past, X_val_fut, y_val
    
    lista = prepare_sequences(SCALER_TYPE, N_PAST) 
    train = lista[0]
    X_train_past, X_train_fut, y_train = train[0], train[1], train[2]
    val = lista[1]
    X_val_past, X_val_fut, y_val = val[0], val[1], val[2]
    
    tuner = kt.Hyperband(
        build_model_for_tuner,
        objective=kt.Objective("val_loss", direction="min"),
        max_epochs=40,
        factor=3,
        directory='my_tuning_dir',
        project_name='fanta_lstm_ranking_v1',
        overwrite=True
    )
    
    stop_early = EarlyStopping(monitor='val_mae', patience=5)
    
    tuner.search(
        [X_train_past, X_train_fut], y_train,
        validation_data=([X_val_past, X_val_fut], y_val),
        epochs=40,
        batch_size=64,
        callbacks=[stop_early],
        verbose=1
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"""
    ✅ TUNING COMPLETATO! I Migliori Parametri sono:
    ------------------------------------------------
    LSTM Units:        {best_hps.get('lstm_units')}
    Dense Context:     {best_hps.get('dense_ctx_units')}
    Dense Final:       {best_hps.get('dense_final_units')}
    Extra Layer?:      {best_hps.get('use_extra_dense_layer')}
    Dropout Rate:      {best_hps.get('dropout_rate')}
    Learning Rate:     {best_hps.get('learning_rate')}
    Loss Cosine Wgt:   {best_hps.get('loss_weight_cosine')}
    """)
    
    return best_hps

if __name__ == "__main__":
    best_params = run_hyperparameter_tuning()