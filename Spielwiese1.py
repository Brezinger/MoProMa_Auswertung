import pandas as pd

def synchronize_data(alphas, p_stats_oben):
    # Konvertiere die Zeitstempel in datetime-Objekte
    alphas['Time'] = pd.to_datetime(alphas['Time'])
    p_stats_oben['Time'] = pd.to_datetime(p_stats_oben['Time'])
    
    # Berechne die Zeitdifferenz zwischen den Zeitstempeln von alphas und stats
    time_difference = alphas['Time'].iloc[0] - p_stats_oben['Time'].iloc[0]
    p_stats_oben['Adjusted_Time'] = p_stats_oben['Time'] + time_difference
    
    # Führe eine äußere Verknüpfung durch, um alle Zeilen beizubehalten
    merged = pd.merge_asof(alphas, p_stats_oben, left_on='Time', right_on='Adjusted_Time')
    
    # Entferne die zusätzliche Spalte
    merged.drop(columns='Adjusted_Time', inplace=True)
    
    # Extrahiere die synchronisierten Daten in zwei separate Listen zusammen mit den Zeitstempeln
    synchronized_alphas = merged[['Time', 'Alpha_Value']].values.tolist()
    synchronized_p_stats_oben = merged[['Time'] + [col for col in merged.columns if col.startswith('K02_')]].values.tolist()
    
    return synchronized_alphas, synchronized_p_stats_oben





# Beispiel-Datenframes für alphas und stats
# alphas_data = {'Time': ['2023-08-04 21:58:18.855000', '2023-08-04 21:58:19.855000', '2023-08-04 21:58:20.855000'],
#                'Alpha_Value': [10, 20, 30]}
# stats_data = {'Time': ['2023-08-04 21:58:18.683', '2023-08-04 21:58:19.683', '2023-08-04 21:58:20.683'],
#               'K02_1': [15, 25, 35],
#               'K02_2': [16, 26, 36],
#               'K02_3': [17, 27, 37]}

# alphas_df = pd.DataFrame(alphas_data)
# stats_df = pd.DataFrame(stats_data)

# Synchronisiere die Daten
synchronized_alphas, synchronized_p_stats_oben = synchronize_data(alphas, p_stats_oben)

# print("Synchronisierte Alphas:")
# for timestamp, value in synchronized_alphas:
#     print(timestamp, value)
    
# print("Synchronisierte Stats:")
# for data in synchronized_stats:
#     print(data)
