import numpy as np

def calculate_team_strength(df):
    df = df.sort_values(['Team_Name', 'Wk']).copy()

    conditions = [
        (df['Goals'] > df['Goals_Ag']), 
        (df['Goals'] == df['Goals_Ag'])
    ]
    df['Raw_Points'] = np.select(conditions, [3, 1], default=0)
    
    difficulty_multiplier = np.where(df['home'] == 0, 1.2, 1.0)
    df['Adjusted_Points'] = df['Raw_Points'] * difficulty_multiplier
    
    df['xG_Ratio'] = df['xG'] / (df['xG'] + df['xG_Ag'] + 0.01)

    
    df['Form_Points_EMA'] = df.groupby('Team_Name')['Adjusted_Points'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
    df['Form_xG_EMA'] = df.groupby('Team_Name')['xG_Ratio'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
    
    df['Form_Score'] = ((df['Form_Points_EMA'] / 3.6) * 0.60 + (df['Form_xG_EMA'] * 0.40)) * 100

    df['Season_PPG_Current'] = df.groupby('Team_Name')['Raw_Points'].transform(lambda x: x.expanding().mean())
    
    df['Pre_Match_PPG'] = df.groupby('Team_Name')['Season_PPG_Current'].shift(1)
    
    df['Pre_Match_PPG'] = df['Pre_Match_PPG'].fillna(1.35)
    
    df['Status_Score'] = (df['Pre_Match_PPG'] / 3.0) * 100
    
    max_wk = df['Wk'].max()
    df['Season_Progress'] = df['Wk'] / max_wk
    
    df['Weight_Status'] = (df['Season_Progress'] * 0.5).clip(0.1, 0.5)
    df['Weight_Form'] = 1 - df['Weight_Status']
    
    df['Final_Strength_Raw'] = (df['Form_Score'] * df['Weight_Form']) + \
                               (df['Status_Score'] * df['Weight_Status'])
    
    df['Pre_Match_Strength'] = df.groupby('Team_Name')['Final_Strength_Raw'].shift(1)
    
    df['Pre_Match_Strength'] = df['Pre_Match_Strength'].fillna(50.0)

    cols_to_drop = ['Raw_Points', 'Adjusted_Points', 'xG_Ratio', 'Form_Points_EMA', 'Form_xG_EMA', 
                    'Form_Score', 'Season_PPG_Current', 'Pre_Match_PPG', 'Status_Score', 
                    'Season_Progress', 'Weight_Status', 'Weight_Form']
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df
