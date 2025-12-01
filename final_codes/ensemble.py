from best_model_lstm import train_and_test_best_config, prepare_sequences, SCALER_TYPE, N_PAST, mean_absolute_error, spearmanr
from best_model_cnn import train_best_cnn


def run_ensemble_test():
    print("\n===========================================")
    print("🚀 AVVIO TEST ENSEMBLE (LSTM + CNN)")
    print("===========================================\n")
    
    # --- 1. ADDESTRAMENTO LSTM ---
    print(">>> Training BEST LSTM...")
    # Assicurati che train_and_test_best_config ritorni (model, mae)
    lstm_model, lstm_mae = train_and_test_best_config(plot_graphs=False)
    print(f"LSTM MAE: {lstm_mae:.4f}\n")
    
    # --- 2. ADDESTRAMENTO CNN ---
    print(">>> Training BEST CNN...")
    # Assicurati che train_best_cnn ritorni (model, mae)
    cnn_model, cnn_mae = train_best_cnn(plot_graphs=False)
    print(f"CNN MAE: {cnn_mae:.4f}\n")
    
    # --- 3. PREPARAZIONE DATI TEST ---
    # Ci servono i dati di TEST grezzi per fare le predizioni manuali
    # Usiamo prepare_sequences per ottenerli
    data_bundle = prepare_sequences(SCALER_TYPE, N_PAST) 
    test_data = data_bundle[2]
    denorm_factor = data_bundle[3]
    
    X_test_p, X_test_f, y_test = test_data[0], test_data[1], test_data[2]
    
    # --- 4. PREDIZIONI INDIVIDUALI ---
    print(">>> Calcolo predizioni finali...")
    pred_lstm = lstm_model.predict([X_test_p, X_test_f], verbose=0).flatten()
    pred_cnn = cnn_model.predict([X_test_p, X_test_f], verbose=0).flatten()
    
    # --- 5. ENSEMBLE (MEDIA SEMPLICE) ---
    # Puoi provare anche pesi diversi: (0.6 * pred_lstm + 0.4 * pred_cnn)
    pred_ensemble = (pred_lstm + pred_cnn) / 2.0
    
    # --- 6. VALUTAZIONE ENSEMBLE ---
    mae_ens_scaled = mean_absolute_error(y_test, pred_ensemble)
    mae_ens_real = mae_ens_scaled * denorm_factor
    spearman_ens, _ = spearmanr(y_test, pred_ensemble)
    
    print("\n===========================================")
    print("🏆 RISULTATI FINALI ENSEMBLE")
    print("===========================================")
    print(f"MAE Reale Ensemble: {mae_ens_real:.4f}")
    print(f"Spearman Ensemble:  {spearman_ens:.4f}")
    
    # --- 7. CONFRONTO DIRETTO ---
    best_single_mae = min(lstm_mae, cnn_mae)
    improvement = best_single_mae - mae_ens_real
    
    if improvement > 0:
        print(f"\n✅ SUCCESSO: L'Ensemble migliora il miglior modello singolo di {improvement:.4f} punti!")
    else:
        print(f"\n⚠️ PECCATO: L'Ensemble non migliora il modello singolo (Overfitting o modelli troppo simili).")
        
    return pred_ensemble

if __name__ == "__main__":
    run_ensemble_test()