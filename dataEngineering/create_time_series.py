import pandas as pd

def extract_player(df):

    df = df.sort_values(by=['Player_id', 'Week_id'])
    df = df.reset_index(drop=True)

    presenze_per_giocatore = df.groupby('Player_id')['Voto_Fantacalcio'].count()

    soglia_minima = 15
    giocatori_validi = presenze_per_giocatore[presenze_per_giocatore >= soglia_minima].index

    df_filtered = df[df['Player_id'].isin(giocatori_validi)].copy()

    time_series_matrix = df_filtered.pivot_table(
    index='Name',           # O 'Player_id'
    columns='Week_id', 
    values='Voto_Fantacalcio'
    )

    print(time_series_matrix.iloc[:5, :10])

