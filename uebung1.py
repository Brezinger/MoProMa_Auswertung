# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:21:04 2024

@author: Besitzer
"""
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta

p_ges_ref= 100500 #erstmal zum manuell Eingeben, später evtl. Zugriff auf Pitot über zwei zugewiesene Druckports
p_stat_ref=98000
t=700 # Profiltiefe
start_time = "2023-08-04 21:58:36"
end_time = "2023-08-04 21:58:37"
#recording_start_time = "2023-08-04 21:58:37"



def log_drive_file(file_path_drive): # Funktion zum Einlesen der drive Datei
    dates = []  # Liste zum Speichern der Daten
    times = []  #absolute Zeit
    positions = [] #Position des Nachlaufrechens
    velocities = [] #Geschwindigkeit des Nachlaufrechens

    with open(file_path_drive, 'r') as file:
        num_entries = sum(1 for line in file)  # Anzahl der Zeilen in der Datei zählen
        file.seek(0)  # Zurück zum Anfang der Datei

        for _ in range(num_entries):
            parts = file.readline().strip().split()  # Zeile in Bestandteile aufteilen
            if len(parts) == 4: # kontrolliert ob Werte richtig eingetragen wurden
                # Datum, Zeit, Position und Geschwindigkeit extrahieren
                date = parts[0]
                time = parts[1]
                position = float(parts[2])
                velocity = float(parts[3])
                #hier wäre ein else super, welches die Zeile rausschmeißt wenn fehlerhaft

                # Daten speichern
                dates.append(date)
                times.append(time)
                positions.append(position)
                velocities.append(velocity)

    return dates, times, positions, velocities

def log_AOA_file(file_path_AOA):
    data = []   # Liste zum Speichern von (time, alpha) Paaren

    with open(file_path_AOA, 'r') as file:
        num_entries = sum(1 for line in file)  # Anzahl der Zeilen in der Datei zählen
        file.seek(0)  # Zurück zum Anfang der Datei

        for _ in range(num_entries):
            parts = file.readline().strip().split()  # Zeile in Bestandteile aufteilen
            if len(parts) == 4:
                # Datum, Zeit, Position und Geschwindigkeit extrahieren
                date = parts[0]
                time = parts[1]
                position = float(parts[2])
                turn = float(parts[3])
                
                # Daten umrechnen nach MoProMa_Lab(-->deg)
                abs_sensor_pos_deg = - position / 2 ** 14 * 360 - turn * 360 + 162.88330078125
                gear_ratio = 60 / (306 * 2)
                alpha = abs_sensor_pos_deg * gear_ratio #degree

                # Daten speichern
                # Zeit und Datum zu einem Pandas Timestamp zusammenfassen
                datetime = pd.to_datetime(f"{date} {time}")
                data.append((datetime, alpha))


    # DataFrame für times und alphas erstellen
    alphas = pd.DataFrame(data, columns=['Time', 'Alpha'])

    return alphas

def calc_meanAOA(alphas, start_time, end_time):
    # Alpha-Werte zwischen den Zeiten start_time und end_time auswählen
    selected_alphas = alphas.loc[(alphas['Time'] >= start_time) & (alphas['Time'] <= end_time), 'Alpha']
    
    # Mittelwert der ausgewählten Alpha-Werte berechnen
    alpha_mean = selected_alphas.mean()
    
    return alpha_mean

def read_DLR_pressure_scanner_file(filename, n_sens, t0): #Liest die Drücke aus den Statikmessstellen auf der Oberseite des Profils ein
    """
    Converts raw sensor data to pandas DataFrame
    :param filename:            File name
    :param n_sens:              number of sensors
    :param t0:                  time of first timestamp
    :return:                    pandas DataFrame with absolute time and pressures
    """

    # Convert start time to milliseconds since it is easier to handle arithmetic operations
    start_time_ms = t0.timestamp() * 1000

    namelist = filename.rstrip(".dat").split("_")
    unit_name = "_".join(namelist[-2:])

    # Spaltennamen für das DataFrame erstellen
    columns = ["Time"] + [unit_name + f"_{i}" for i in range(1, n_sens+1)]

    # Einlesen der Daten in ein DataFrame
    df = pd.read_csv(filename, sep="\s+", header=None, names=columns, usecols=range(n_sens+1), on_bad_lines='skip',
                     engine='python')

    # drop lines with missing data
    df = df.dropna().reset_index(drop=True)

    # Calculate the time difference in milliseconds from the first row
    time_diff_ms = df['Time'] - df['Time'].iloc[0]

    # Add this difference to the start time (in milliseconds) and convert back to datetime
    df['Time'] = pd.to_datetime(start_time_ms + time_diff_ms, unit='ms')
                
    return df

def log_pstat_unten_file(file_path_stat_unten): #Liest die Drücke aus den Statikmessstellen auf der Unterseite des Profils ein
        # Einlesen der ersten Zeile, um die Anzahl der Spalten zu bestimmen
        with open(file_path_stat_unten, 'r') as file:
            first_line = file.readline().split()

        # Die Anzahl der Spalten ohne die erste und letzte Spalte bestimmen
        anzahl_spalten = len(first_line) - 2

        # Spaltennamen für das DataFrame erstellen
        spaltennamen = ["Time"] + [f"K03_{i}" for i in range(1, anzahl_spalten + 1)]

        # Einlesen der Daten in ein DataFrame
        p_stats_unten = pd.read_csv(file_path_stat_unten, sep="\s+", header=None, names=spaltennamen, usecols=range(anzahl_spalten + 1), engine='python')

        # Die letzte Spalte löschen
        p_stats_unten.drop(columns=["K03_" + str(anzahl_spalten)], inplace=True)
                    
        return p_stats_unten

def adjust_timestamps(alphas, p_stats_oben):
    # Extrahiere den Startzeitstempel des ersten Datensatzes (als pandas Timestamp oder String)
    start_timestamp_str = alphas.iloc[0, 0]
    
    # Konvertiere start_timestamp_str in ein datetime-Objekt, falls es nicht bereits eines ist
    if isinstance(start_timestamp_str, pd.Timestamp):
        start_timestamp = start_timestamp_str.to_pydatetime()
    else:
        start_timestamp = datetime.strptime(start_timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    
    # Extrahiere den Startwert in Millisekunden für den zweiten Datensatz aus p_stats_oben
    start_milliseconds = p_stats_oben.iloc[0, 0]
    
    # Konvertiere den Startwert in Millisekunden für den zweiten Datensatz in einen Zeitstempel
    start_milliseconds_timestamp = datetime.fromtimestamp(start_milliseconds / 1000)
    
    # Berechne die Differenz in Millisekunden zwischen den beiden Startwerten
    diff_ms = start_timestamp - start_milliseconds_timestamp
    
    # Passe die Zeitstempel des zweiten Datensatzes an
    for i, ms in enumerate(p_stats_oben.iloc[:, 0]):
        adjusted_time = start_timestamp + timedelta(milliseconds=int(ms) + diff_ms.total_seconds() * 1000)
        p_stats_oben.iloc[i, 0] = adjusted_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Benenne die Spalte "adjusted_time" in "time" um
    p_stats_oben.rename(columns={p_stats_oben.columns[0]: 'time'}, inplace=True)
    
    return p_stats_oben







def synchronize_data(merge_dfs_list):
    """
    Syncronizes and interpolates sensor data, given in pandas DataFrames with a timestamp
    :param merge_dfs_list:      list of pandas DataFrames containing sensor data. Must contain "Time" column in
                                datetime format
    :return:                    merged dataframe with all sensor data, interpolated according time
    """

    # Merge the DataFrames using merge_asof
    merged_df = merge_dfs_list[0]
    for df in merge_dfs_list[1:]:
        merged_df = pd.merge_asof(merged_df, df, on='Time', tolerance=pd.Timedelta('1ms'), direction='nearest')

    # Set the index to 't_abs' to use time-based interpolation
    merged_df.set_index('Time', inplace=True)

    # Interpolate missing values using time-based interpolation
    interpolated_df = merged_df.interpolate(method='time')

    return interpolated_df

# # Beispielaufruf der Funktion
# # Beispiel-Daten erstellen
# alphas = pd.DataFrame({
#     'Time': pd.date_range(start='2023-08-04 21:58:18', periods=6, freq='S'),
#     'Alpha': [2.996467, 2.996467, 2.996467, 2.996467, 2.996467, 2.996467]
# })

# p_stats_oben = pd.DataFrame({
#     'Time': pd.date_range(start='2023-08-04 21:58:20', periods=6, freq='S'),
#     'Pressure': [200, 300, 400, 500, 600, 700]
# })

# synchronized_alphas, synchronized_p_stats_oben = synchronize_data(alphas, p_stats_oben)












# def synchronize_data(alphas, p_stats_oben):
    
    
#     # Konvertiere die Zeitstempel in datetime-Objekte
#     alphas['Time'] = pd.to_datetime(alphas['Time'], unit='s')
#     p_stats_oben['Time'] = pd.to_datetime(p_stats_oben['Time'], unit='s')
#     print(p_stats_oben)
    
#     # Führe eine äußere Verknüpfung durch
#     merged = pd.merge(alphas, p_stats_oben, on='Time', how='outer')
#     #print(merged)
    
#     # Interpoliere fehlende Werte
#     merged.interpolate(method='linear', inplace=True)
    
#     # Entferne die zusätzlichen Spalten
#     synchronized_alphas = merged[['Time', 'Alpha']]
#     synchronized_p_stats_oben = merged.drop(columns=['Alpha'])
    
#     return synchronized_alphas, synchronized_p_stats_oben




# def berechne_cpi(p_stats_oben, p_ges_ref, p_stat_ref): #erstellt ein dict in dem für jede Messstelle
#     #ein cp Wert in abhängigkeit der Zeit berechnet wird
    
#     cpi_dict = {}
#     for key, values in p_stats_oben.items():
#         cpi_values = []
#         for i, value in enumerate(values):
#             cpi = 1 - ((p_ges_ref - value) / (p_ges_ref - p_stat_ref))
#             cpi_values.append(cpi)
#         cpi_dict['cp' + key[1:]] = cpi_values
        
#     return cpi_dict

# def log_airfoil_file(file_path_airfoil):
#     # Excel-Datei einlesen
#     daten = pd.read_excel(file_path_airfoil)
    
#     # x-Koordinaten
#     # Klappe oben: B3 bis B9 auswählen und in einen Vektor speichern
#     x_Kl_o = daten.iloc[1:7, 1].tolist()
#     # Klappe unten: B42 bis B61 auswählen und in einen Vektor speichern
#     x_Kl_u = daten.iloc[60:64, 1].tolist()
#     # Flügel oben:B10...B40 ohne B28, B40 auswählen und in einen Vektor speichern
#     x_Fl_o_1 = daten.iloc[8:26, 1].tolist()
#     x_Fl_o_2 = daten.iloc[27:38, 1].tolist()
#     x_Fl_o = x_Fl_o_1 + x_Fl_o_2
#     # Flügel unten: B10 bis B40 auswählen und in einen Vektor speichern
#     x_Fl_u = daten.iloc[40:60, 1].tolist()
#     x_o=x_Fl_o + x_Kl_o
#     x_u=x_Fl_u + x_Kl_u
    
    
#     # y-Koordinaten
#     # Klappe oben: C3 bis C9 auswählen und in einen Vektor speichern
#     y_Kl_o = daten.iloc[1:7, 2].tolist()
#     # Klappe unten: C62 bis C66 auswählen und in einen Vektor speichern
#     y_Kl_u = daten.iloc[60:64, 2].tolist()
#     # Flügel oben:C10...C40 ohne C28, C40 auswählen und in einen Vektor speichern
#     y_Fl_o_1 = daten.iloc[8:26, 2].tolist()
#     y_Fl_o_2 = daten.iloc[27:38, 2].tolist()
#     y_Fl_o = y_Fl_o_1 + y_Fl_o_2
#     # Flügel unten: C42 bis C61 auswählen und in einen Vektor speichern
#     y_Fl_u = daten.iloc[40:60, 2].tolist()
#     y_o = y_Fl_o + y_Kl_o
#     y_u = y_Fl_u + y_Kl_u
    
    
#     return x_o, x_u, y_o, y_u

# def calc_cn_ct(cpi_dict, x_o, y_o):
#     cn = []
#     ct = []
#     keys = sorted(cpi_dict.keys(), key=lambda x: int(x[2:]))  # Sortiert die Schlüssel nach der Zahl in "cpX" (=Liste)
#     # i... Messungen (zeitabhängig)
#     # j... Messstellen (immer gleich für Profil)
#     for i in range(len(cpi_dict[keys[0]])):  # Iteriert über die Länge einer der Listen (Anzahl an Messungen)
#         cn_i = 0
#         for j in range(1, len(keys)):  # Iteriert über die Anzahl der Messtellen
#             cn_i += ((cpi_dict[keys[j]][i] + cpi_dict[keys[j-1]][i]) / 2) * ((x_o[j] - x_o[j-1]) / t)
#         cn.append(cn_i)
        
#     for i in range(len(cpi_dict[keys[0]])):  # Iteriert über die Länge einer der Listen (Anzahl an Messungen)
#         ct_i = 0
#         for j in range(1, len(keys)):  # Iteriert über die Anzahl der Messtellen
#             ct_i += ((cpi_dict[keys[j]][i] + cpi_dict[keys[j-1]][i]) / 2) * ((y_o[j] - y_o[j-1]) / t)
#         ct.append(ct_i)   
        
        
#     return cn, ct

# def calc_ca_cw(cn,ct,alpha_mean):
#     ca=[]
#     cw=[]
    
#     for i in range(1, len(cn)):
#         cai=cn[i]*math.cos(alpha_mean*math.pi/180)-ct[i]*math.sin(alpha_mean*math.pi/180)
#         cwi=cn[i]*math.sin(alpha_mean*math.pi/180)-ct[i]*math.cos(alpha_mean*math.pi/180)
#         ca.append(cai)
#         cw.append(cwi)
        
#     return ca, cw



if __name__ == '__main__':
    file_path_drive = '20230804-235819_drive.dat'
    file_path_AOA = '20230804-235818_AOA.dat'
    file_path_stat_oben = '20230804-235818_static_K02.dat'
    file_path_stat_unten = '20230804-235818_static_K03.dat'
    file_path_airfoil = 'airfoil_geometry_Mu_13_33.xlsx'

    dates, times, positions, velocities = log_drive_file(file_path_drive)
    alphas = log_AOA_file(file_path_AOA)
    alpha_mean = calc_meanAOA(alphas, start_time, end_time)
    p_stats_oben = read_DLR_pressure_scanner_file(file_path_stat_oben, n_sens=32, t0=alphas["Time"].iloc[0])
    p_stats_unten = read_DLR_pressure_scanner_file(file_path_stat_unten, n_sens=32, t0=alphas["Time"].iloc[0])
    df_sync = synchronize_data([p_stats_oben, p_stats_unten, alphas])
    # cpi_dict = berechne_cpi(p_stats_oben, p_ges_ref, p_stat_ref)
    # x_o, x_u, y_o, y_u = log_airfoil_file(file_path_airfoil)
    # cn, ct = calc_cn_ct(cpi_dict, x_o, y_o)
    # ca, cw = calc_ca_cw(cn,ct,alpha_mean)
    print('done')




