import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])

    cols_to_exclude = ['Player_id', 'Week_id', 'game_id', 'Team_id', 'Opponent_id']
    numeric_df = numeric_df.drop(columns=cols_to_exclude, errors='ignore')

    corr_matrix = numeric_df.corr().abs()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("Matrice di Correlazione tra Statistiche")
    plt.show()

    threshold = 0.75

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Feature troppo correlate da eliminare ({len(to_drop)}):")
    print(to_drop)


def drop_useless_features(df):
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
