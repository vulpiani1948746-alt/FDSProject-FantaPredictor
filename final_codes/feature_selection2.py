import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from lstm_with_predictions import prepare_sequences, player_cols, team_past_cols, team_future_cols
from cnn import train_and_test_cnn


def calculate_permutation_importance(model, X_past, X_fut, y_true, 
                                     past_col_names, fut_col_names):
    """
    Calculates Permutation Feature Importance for a dual-input model (Past + Future).
    """
    
    # BASELINE
    print("Compute Baseline")
    preds_baseline = model.predict([X_past, X_fut], verbose=0).flatten()
    baseline_mae = mean_absolute_error(y_true, preds_baseline)
    print(f"Baseline MAE: {baseline_mae:.5f}\n")
    
    results = []
    
    # ANALYZE PAST FEATURES (Historical Stats)
    print("Feature Analysis...")
    for i, col_name in enumerate(past_col_names):
        X_past_shuffled = X_past.copy()
        
        # Shuffle the i-th feature across all samples
        idx = np.random.permutation(X_past.shape[0])
        X_past_shuffled[:, :, i] = X_past_shuffled[idx, :, i]
        
        # Predict with the corrupted dataset
        preds_shuffled = model.predict([X_past_shuffled, X_fut], verbose=0).flatten()
        mae_shuffled = mean_absolute_error(y_true, preds_shuffled)
        
        # Calculate Importance (Drop in performance)
        importance = mae_shuffled - baseline_mae
        results.append({'Feature': col_name, 'Importance': importance, 'Type': 'Past'})
        print(f"  -> {col_name}: +{importance:.5f}")

    # ANALYZE FUTURE FEATURES 
    print("\nAnalisi feature future (Context)...")
    for i, col_name in enumerate(fut_col_names):
        X_fut_shuffled = X_fut.copy()
        
        np.random.shuffle(X_fut_shuffled[:, i])
        
        preds_shuffled = model.predict([X_past, X_fut_shuffled], verbose=0).flatten()
        mae_shuffled = mean_absolute_error(y_true, preds_shuffled)
        
        importance = mae_shuffled - baseline_mae
        results.append({'Feature': col_name, 'Importance': importance, 'Type': 'Future'})
        print(f"  -> {col_name}: +{importance:.5f}")

    # Return results sorted by importance (highest first)
    importance_df = pd.DataFrame(results).sort_values(by='Importance', ascending=False)
    return importance_df


def compute_importances():
    """
    Orchestrator function to:
    1. Train a model.
    2. Compute feature importance on validation data.
    3. Plot results.
    4. Filter out useless features.
    """
    # Train the model
    model = train_and_test_cnn()[0]
    
    # Prepare data sequences
    lista = prepare_sequences()

    # Use Validation Set for importance calculation to avoid overfitting to Train or leaking Test data
    val = lista[1]
    X_val_past = val[0]
    X_val_fut = val[1]
    y_val = val[2]

    # Combine names for tracking
    names_past = player_cols + team_past_cols
    names_fut = team_future_cols

    # Calculate Importance
    imp_df = calculate_permutation_importance(
        model, 
        X_val_past, X_val_fut, y_val,
        names_past, names_fut
    )
    
    # Visualization
    plot_importance(imp_df)

    # Filter features: Remove any feature with importance <= 0.0
    player_cols_new, team_past_cols_new, team_future_cols_new = filter_features_by_importance(
    imp_df, 
    player_cols, 
    team_past_cols, 
    team_future_cols, 
    threshold=0.0
    )

    return player_cols_new, team_past_cols_new, team_future_cols_new


def plot_importance(importance_df):
    """
    Visualizes feature importance using a horizontal bar chart.
    Features are colored by type (Past/Future).
    """
    plt.figure(figsize=(10, 8))
    
    # Map colors: Blue for historical data, Orange for future context
    colors = importance_df['Type'].map({'Past': 'skyblue', 'Future': 'orange'})
    
    plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    plt.xlabel('Increase in MAE (Higher = More Important)')
    plt.title('Feature Importance (Permutation)')
    
    # Invert Y axis to have the most important feature at the top
    plt.gca().invert_yaxis()
    
    # Add a red line at 0: Features to the left are useless/harmful
    plt.axvline(x=0, color='red', linestyle='--')
    plt.grid(axis='x', alpha=0.3)
    
    plt.show()


def filter_features_by_importance(importance_df, player_cols, team_past_cols, team_future_cols, threshold=0.0):
    """
    Removes features with importance below the threshold.
    Updates the global configuration lists for the next training run.
    """
    # Identify features to drop
    features_to_drop = importance_df[importance_df['Importance'] <= threshold]['Feature'].tolist()
    
    print(f"--- FEATURE SELECTION ---")
    print(f"Cutoff Threshold: {threshold}")
    print(f"Dropped Features ({len(features_to_drop)}): {features_to_drop}")
    
    # Rebuild lists excluding dropped features
    new_player_cols = [c for c in player_cols if c not in features_to_drop]
    new_team_past_cols = [c for c in team_past_cols if c not in features_to_drop]
    new_team_future_cols = [c for c in team_future_cols if c not in features_to_drop]
    
    # Summary of changes
    print(f"\nPlayer cols: {len(player_cols)} -> {len(new_player_cols)}")
    print(f"Team Past cols: {len(team_past_cols)} -> {len(new_team_past_cols)}")
    print(f"Team Future cols: {len(team_future_cols)} -> {len(new_team_future_cols)}")
    
    return new_player_cols, new_team_past_cols, new_team_future_cols


if __name__ == "__main__":
    compute_importances()