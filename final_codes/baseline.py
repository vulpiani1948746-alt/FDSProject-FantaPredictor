import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr


def baseline_regression(X_train_p, X_train_f, y_train, X_test_p, X_test_f, y_test, denorm_factor):
    """
    Constructs the baseline model as a simple linear regression 
    and computes the metrics for it
    """
    print("\n--- COMPUTING THE BASELINE (Linear Regression) ---")
    
    # Flattening: (Samples, Timesteps, Features) -> (Samples, Timesteps * Features)
    N_train, T_train, F_train = X_train_p.shape
    X_train_p_flat = X_train_p.reshape(N_train, T_train * F_train)
    X_train_combined = np.concatenate([X_train_p_flat, X_train_f], axis=1)
    
    N_test, T_test, F_test = X_test_p.shape
    X_test_p_flat = X_test_p.reshape(N_test, T_test * F_test)
    X_test_combined = np.concatenate([X_test_p_flat, X_test_f], axis=1)
    
    # Training
    regressor = LinearRegression()
    regressor.fit(X_train_combined, y_train)
    
    # Prediction
    preds = regressor.predict(X_test_combined)
    
    # Metrics
    mae_scaled = mean_absolute_error(y_test, preds)
    mae_real = mae_scaled * denorm_factor
    spearman_corr, _ = spearmanr(y_test, preds)
    
    print(f"MAE Linear Regression: {mae_real:.4f}")
    print(f"Spearman Correlation:  {spearman_corr:.4f}")
    
    return mae_real, spearman_corr, regressor, preds
