import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def correlation_matrix(df):
    """
    Computes and plots the correlation matrix; identifies features with correlation > 0.75.
    """
    # Select numeric features and exclude non-predictive IDs
    numeric_df = df.select_dtypes(include=[np.number])
    cols_to_exclude = ['Player_id', 'Week_id', 'game_id', 'Team_id', 'Opponent_id']
    numeric_df = numeric_df.drop(columns=cols_to_exclude, errors='ignore')

    # Compute absolute correlation
    corr_matrix = numeric_df.corr().abs()

    # Visualize heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

    # Find highly correlated features using the upper triangle to avoid duplicates
    threshold = 0.75
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # List features that exceed the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Highly correlated features to drop ({len(to_drop)}):")
    print(to_drop)


def drop_useless_features(df):
    """
    Removes specific columns deemed redundant or potential data leakage.
    """
    cols_to_drop = [
        "PKatt",          
        "Player_npxG",    
        "Cmp_x",         
        "Att_x",
        "Carries",
        "Voto_Statistico", 
        "Voto_Italia"
    ]

    df = df.drop(columns=cols_to_drop, errors='ignore')

    return df