from best_model_lstm import train_and_test_best_config, prepare_sequences, SCALER_TYPE, N_PAST, mean_absolute_error, spearmanr
from best_model_cnn import train_best_cnn


def run_ensemble_test():
    """
    Test a model that ensembles the predictions of the best LSTM and the best CNN
    """
    print("TEST ENSEMBLE (LSTM + CNN)")
    print("===========================================\n")
    
    print(">>> Training BEST LSTM...")
    lstm_model, lstm_mae = train_and_test_best_config(plot_graphs=False)
    print(f"LSTM MAE: {lstm_mae:.4f}\n")
    
    print(">>> Training BEST CNN...")
    cnn_model, cnn_mae = train_best_cnn(plot_graphs=False)
    print(f"CNN MAE: {cnn_mae:.4f}\n")
    
    # Test data preparation
    data_bundle = prepare_sequences(SCALER_TYPE, N_PAST) 
    test_data = data_bundle[2]
    denorm_factor = data_bundle[3]
    
    X_test_p, X_test_f, y_test = test_data[0], test_data[1], test_data[2]
    
    print(">>> Computing final predictions...")
    pred_lstm = lstm_model.predict([X_test_p, X_test_f], verbose=0).flatten()
    pred_cnn = cnn_model.predict([X_test_p, X_test_f], verbose=0).flatten()
    
    # Ensemble
    pred_ensemble = (pred_lstm + pred_cnn) / 2.0
    
    # Final Evaluation
    mae_ens_scaled = mean_absolute_error(y_test, pred_ensemble)
    mae_ens_real = mae_ens_scaled * denorm_factor
    spearman_ens, _ = spearmanr(y_test, pred_ensemble)
    
    print("\n===========================================")
    print("ENSEMBLE FINAL RESULTS")
    print("===========================================")
    print(f"MAE Reale Ensemble: {mae_ens_real:.4f}")
    print(f"Spearman Ensemble:  {spearman_ens:.4f}")
    
    # Comparison
    best_single_mae = min(lstm_mae, cnn_mae)
    improvement = best_single_mae - mae_ens_real
    
    if improvement > 0:
        print(f"\nSUCCESS: The ensemble improves the model by {improvement:.4f} points!")
    else:
        print(f"\nUNSUCCESS: The ensemble does not improve the model.")
        
    return pred_ensemble

if __name__ == "__main__":
    run_ensemble_test()