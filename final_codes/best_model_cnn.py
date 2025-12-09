import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from lstm_with_predictions import prepare_sequences, get_ranking_loss, plot_diagnostics
from lstm_with_predictions import N_PAST, M_FUTURE, SCALER_TYPE 


def build_best_cnn(shape_past, shape_future):
    """
    Builds the Best model for the CNN by manually inserting the hyperparameters
    obtained during the tuning.
    """
    # Parameters obtained during the tuning
    FILTERS_L1 = 96
    FILTERS_L2 = 194         
    KERNEL_SIZE = 3
    DENSE_CONTEXT = 32
    DENSE_FINAL = 64
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    COSINE_WEIGHT = 12.0
    
    # Input
    input_past = Input(shape=shape_past, name='Input_Storico_CNN')
    input_future = Input(shape=shape_future, name='Input_Contesto')
    
    # CNN Part (for the past)
    
    # First Layer
    x = Conv1D(filters=FILTERS_L1, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(input_past)
    x = BatchNormalization()(x)
    
    # Second Layer
    x = Conv1D(filters=FILTERS_L2, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Pooling
    x = GlobalAveragePooling1D()(x) 
    x = Dropout(DROPOUT_RATE)(x)
    
    # Dense Layer
    y = Dense(DENSE_CONTEXT, activation='relu')(input_future)
    y = BatchNormalization()(y)
    y = Dropout(DROPOUT_RATE)(y)
    
    combined = Concatenate()([x, y])
    
    # Final Layer
    z = Dense(DENSE_FINAL, activation='relu')(combined)
    z = Dropout(DROPOUT_RATE)(z)
    
    output = Dense(1, activation='linear', name='Output_Voto')(z)
    
    model = Model(inputs=[input_past, input_future], outputs=output)
    
    loss_ranking = get_ranking_loss(weight_mse=1.0, weight_cos=COSINE_WEIGHT)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='mae', 
                  metrics=['mae', 'mse'])
    
    return model


def train_best_cnn(plot_graphs=True):
    """
    Trains and tests the best possible model for the CNN returning the model and the metrics
    """
    print("\nSTART CNN TRAINING FOR FINAL MODEL")
    
    # Data Preparation
    data_bundle = prepare_sequences(SCALER_TYPE, N_PAST, M_FUTURE)
    
    # Unpack
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


    shape_p = (X_train_p.shape[1], X_train_p.shape[2])
    shape_f = (X_train_f.shape[1],)
    
    model = build_best_cnn(shape_p, shape_f)
    model.summary()

    # Training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1)
    ]

    # Fitting the model
    history = model.fit(
        x=[X_train_p, X_train_f], 
        y=y_train,
        validation_data=([X_val_p, X_val_f], y_val),
        epochs=200,    
        batch_size=64,   
        callbacks=callbacks,
        verbose=1 
    )

    print("TRAINING DONE.")

    # Plotting
    if plot_graphs:
        try:
            plot_diagnostics(history, model, X_test_p, X_test_f, y_test, denorm_factor)
        except Exception as e:
            print(f"Impossibile generare grafico: {e}")

    # Final Evaluation
    print("\n--- RISULTATI FINALI CNN ---")
    preds = model.predict([X_test_p, X_test_f], verbose=0).flatten()
    
    mae_scaled = mean_absolute_error(y_test, preds)
    mae_real = mae_scaled * denorm_factor
    spearman, _ = spearmanr(y_test, preds)
    
    print(f"MAE Reale: {mae_real:.4f}")
    print(f"Spearman:  {spearman:.4f}")
    
    return model, mae_real


def find_best_n_past_cnn(min_n=3, max_n=12):
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
            
            # Constructs the model
            shape_p = (X_train_p.shape[1], X_train_p.shape[2])
            shape_f = (X_train_f.shape[1],)
            
            model = build_best_cnn(shape_p, shape_f)
            
            # Training
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

    # Final result
    best_n = max(results_spearman, key=results_spearman.get)
    print("\n=========================================")
    print(f"🏆 WINNWER CNN (Spearman): N_PAST = {best_n}")
    print(f"   Spearman: {results_spearman[best_n]:.4f}")
    print(f"   MAE:      {results_mae[best_n]:.4f}")
    print("=========================================")
    
    # Plot to compare
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('N_PAST')
    ax1.set_ylabel('MAE', color=color)
    ax1.plot(list(results_mae.keys()), list(results_mae.values()), marker='o', color=color, label='MAE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Spearman', color=color)
    ax2.plot(list(results_spearman.keys()), list(results_spearman.values()), marker='s', linestyle='--', color=color, label='Spearman')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Performance CNN with varying N_PAST: MAE vs Spearman')
    fig.tight_layout()  
    plt.show()
    
    return best_n, results_spearman


if __name__ == "__main__":
    best_n_lstm, _ = find_best_n_past_cnn(min_n=2, max_n=8)