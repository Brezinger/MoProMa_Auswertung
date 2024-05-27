# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:21:04 2024

@author: Besitzer
"""
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import interpolate, integrate, optimize, signal

start_time = "2023-08-04 21:58:19"
end_time = "2023-08-04 21:58:49"
l_ref = 700 #[mm]



def read_AOA_file(file_path_AOA):
    """
    Converts raw AOA data to pandas DataFrame
    :param file_path_AOA:       File name
    :return: alphas             pandas DataFrame with AOA values
    """
    # Read the entire file into a pandas DataFrame
    df = pd.read_csv(file_path_AOA, sep='\s+', header=None, names=['Date', 'Time', 'Position', 'Turn'])
    # Filter out rows that do not have exactly 4 parts (though this should not happen with the current read_csv)
    df = df.dropna()

    # Compute the absolute sensor position in degrees
    abs_sensor_pos_deg = - df['Position'] / 2**14 * 360 - df['Turn'] * 360 + 162.88330078125
    # Compute the gear ratio and alpha
    gear_ratio = 60 / (306 * 2)
    df['Alpha'] = abs_sensor_pos_deg * gear_ratio

    # Combine Date and Time into a single pandas datetime column
    df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    # Select only the relevant columns
    alphas = df[['Time', 'Alpha']]

    return alphas

def calc_mean(df_raw, start_time, end_time):
    """
    Calculates the mean value of every column in the specified time interval
    :param filename:            File name
    :param start_time:          begin of interval
    :param end_time:            time of first interval
    :return:                    pandas DataFrame with absolute time and pressures
    """
    # Select rows between start_time and end_time
    selected_values = df_raw.loc[(df_raw['Time'] >= start_time) & (df_raw['Time'] <= end_time)]
    
    # Calculate the mean value for every column except 'Time'
    mean_values = selected_values.drop(columns=['Time']).mean()
    mean_value = float(mean_values.iloc[0])
    
    return mean_value

def read_DLR_pressure_scanner_file(filename, n_sens, t0):
    """
    Converts raw sensor data to pandas DataFrame
    :param filename:            File name
    :param n_sens:              number of sensors
    :param t0:                  time of first timestamp
    :return:                    pandas DataFrame with absolute time and pressures
    """

    # Convert start time to milliseconds since it is easier to handle arithmetic operations
    start_time_ms = t0.timestamp() * 1000

    # usual filename: "20230804-235818_static_K0X.dat"; drops .dat and splits name at underscores
    namelist = filename.rstrip(".dat").split("_")
    # generates base for column name of sensors; pattern: static_K0X_Y
    unit_name = "_".join(namelist[-2:])

    # generates final column name for DataFrame (time and static pressure sensor)
    columns = ["Time"] + [unit_name + f"_{i}" for i in range(1, n_sens+1)]

    # loading data into DataFrame
    df = pd.read_csv(filename, sep="\s+", header=None, names=columns, usecols=range(n_sens+1), on_bad_lines='skip',
                     engine='python')

    # drop lines with missing data
    df = df.dropna().reset_index(drop=True)

    # Calculate the time difference in milliseconds from the first row
    time_diff_ms = df['Time'] - df['Time'].iloc[0]

    # Add this difference to the start time (in milliseconds) and convert back to datetime
    df['Time'] = pd.to_datetime(start_time_ms + time_diff_ms, unit='ms')
                
    return df

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

def read_airfoil_geometry(filename, n_mp):
    """
    -> generates a DataFrame with the x and y positions of the measuring points from Excel
    -> adds assignment of sensor unit + port to measuring point from Excel
    -> adds inop information from Excel
    -> drops measuring points with inop status and resets the index
    -> renames last two columns for pstat and ptot assignment (always last two rows in Excel)
    -> generates a new row with sensor names as labeled in column index of df_sync
    :param filename:            file name of Excel eg. "Messpunkte Demonstrator.xlsx".
    :param n_mp:                number of measuring points on airfoil (inop sensors inclusively)
    :return df_airfoil:         DataFrame with info described above
    """
    
    # name of the columns (mp=measuring point)
    columns = ['mp' + f"_{i}" for i in range(1, n_mp+1)]
    
    # Read column C for x-Positions
    df_x = pd.read_excel(filename, usecols="C", skiprows=3, nrows=97, header=None)# Read the Excel file from C2 to C99
    df_x = df_x.T # Transpose the data to make it a single row
    df_x.columns = columns # Rename the columns to match the layout of df_sync
    df_x.index = ['x [mm]'] # Rename index
    
    # Read column D for y-Positions
    df_y = pd.read_excel(filename, usecols="D", skiprows=3, nrows=97, header=None)
    df_y = df_y.T
    df_y.columns = columns
    df_y.index = ['y [mm]']
    
    # Read columns E and F for sensor unit and port
    df_unit = pd.read_excel(filename, usecols="E", skiprows=3, nrows=97, header=None).T
    df_unit.columns = columns
    df_unit.index = ['sensor unit']
    df_port = pd.read_excel(filename, usecols="F", skiprows=3, nrows=97, header=None).T
    df_port.columns = columns
    df_port.index = ['port']
    
    # Read column G for inop sensors
    df_inop = pd.read_excel(filename, usecols="G", skiprows=3, nrows=97, header=None).T
    df_inop.columns = columns
    df_inop.index = ['inop sensors']
    
    df_airfoil = pd.concat([df_x, df_y, df_unit, df_port, df_inop])

    # drops sensors with 'inop' status
    df_airfoil = df_airfoil.drop(columns=df_airfoil.columns[df_airfoil.eq('inop').any()])
    num_cols = len(df_airfoil.columns)
    new_columns = ['mp_' + str(i + 1) for i in range(num_cols)] # renames the columns
    df_airfoil.columns = new_columns # assign the new column names
    
    # Rename pstat and ptot column
    column_pstat = df_airfoil.columns[-2]
    column_ptot = df_airfoil.columns[-1]
    df_airfoil.rename(columns={column_pstat:'pstat'}, inplace=True)
    df_airfoil.rename(columns={column_ptot:'ptot'}, inplace=True)

    # generates name in format of df_sync column index in a new row
    new_row = ['static_K0' + str(int(row3)) + '_' + str(int(row4)) for row3, row4 in zip(df_airfoil.iloc[2], df_airfoil.iloc[3])]
    df_airfoil.loc[len(df_airfoil)] = new_row
    df_airfoil.rename(index={df_airfoil.index[-1]: 'name'}, inplace = True) # name the row
    return df_airfoil

def sort_sync_data(df_airfoil, df_sync):
    """
    -> brings columns of df_sync in order of measuring points of airfoil as assigned in df_airfoil 
    -> the last two columns are pstat and ptot of pitotstatic measurement unit
    :param df_airfoil:          any pandas DataFrame containing string names of sensor ports as found in index 
                                of df_sync
    :param df_sync:             any pandas DataFrame containing string names of sensor ports in index and sensor 
                                pressures as values                      
    :return df_sync_sort:       static pressure in order of measuring points of airfoil (reordered, synchronized DataFrame)
    """
    # cuts for this operation useless rows in df_airfoil
    df_airfoil=df_airfoil.iloc[5:]
    # brings static pressure in order of measuring points of airfoil
    df_sync_sort = df_sync.reindex(columns=df_airfoil.values[0])
    
    #rename column index
    mp_names = [f'mp_{i+1}' for i in range(len(df_sync_sort.columns))]
    df_sync_sort.columns = mp_names
    
    # rename pstat and ptot column
    column_pstat = df_sync_sort.columns[-2]
    column_ptot = df_sync_sort.columns[-1]
    df_sync_sort.rename(columns={column_pstat:'pstat'}, inplace=True)
    df_sync_sort.rename(columns={column_ptot:'ptot'}, inplace=True)

    return df_sync_sort

def calc_cp(df_sync_stat_sort):
        """
        calculates for each static port on airfoil pressure coefficient
        :param sync_stat_sort:      synchronized and sorted DataFrame with static pressure data ("static_K0X_Y")
        :return: df_cp              pandas DataFrame with pressure coefficient in "static_K0X_Y" columns for every
                                    measuring point
        """
        # searchs for column names starting with 'mp_' (=static airfoil pressures)
        columns_to_transform = [col for col in df_sync_stat_sort.columns if col.startswith('mp_')]
        
        ptot = df_sync_stat_sort['ptot']
        pstat = df_sync_stat_sort['pstat']
        # Initialize an empty DataFrame with the same index as df_sync_stat_sort
        df_cp = pd.DataFrame(index=df_sync_stat_sort.index)
        # calculates cp values for adressed columns
        df_cp[columns_to_transform] = df_sync_stat_sort[columns_to_transform].apply(lambda x: 1 - (ptot - x) / (ptot - pstat), axis=0)

        return df_cp

def calc_cn_ct(df_cp, df_airfoil, t):
    """
    Calculates normal and tangential (based on airfoil chord) force coefficient. All information at certain 
    measuring points are integrated to one value. Equations from Döller page 40
    :param df_cp:               list of pandas DataFrames containing cp values 
    :param df_airfoil:          list of pandas DataFrames containing rows of x and y positions of measuring points
                                on airfoil
    :return:df_cn_ct            merged dataframe with cn and ct coefficient at certain times
    """
    # Ensure the column indexes match (drops pstat and ptot column)
    df_cp = df_cp.iloc[:,:-2]
    columns = df_cp.columns.intersection(df_airfoil.columns)
    
    # Extract relevant data from both dataframes
    cp = df_cp[columns]
    x = df_airfoil.loc["x [mm]", columns]
    y = df_airfoil.loc["y [mm]", columns]
    # calculate cn
    cn_values = (((cp.shift(-1, axis=1) + cp) / 2) *  ((x.diff().shift(-1)) / t)).sum(axis=1)
    # calculate ct
    ct_values = (((cp.shift(-1, axis=1) + cp) / 2) *  ((y.diff().shift(-1)) / t)).sum(axis=1)
    
    # Create a result dataframe with cn values
    df_cn_ct = pd.DataFrame({'cn': cn_values, 'ct': ct_values})
    
    return df_cn_ct

def calc_cl_cd(df_cn_ct, df_sync):
    """
    Calculates lift and drag coefficient. Drag coefficient derived from static pressure ports on airfoil!
    Equations from Döller page 41, applying wind tunnel correction according Althasus eq. 36
    :param df_cn_ct:        list of pandas DataFrames containing "cn" and "ct" column
    :param df_sync:         list of pandas DataFrames containing column "Alpha"
    :return: df_cl_cd:      merged dataframe with lift and drag coefficient at certain times
    """
   
    alpha = df_sync.loc[:, "Alpha"] # extracting the syncronized alpha column from df_sync
    cn = df_cn_ct.loc[:,"cn"]  # extracting the cn column from df_cn_ct
    ct = df_cn_ct.loc[:,"ct"]  # extracting the ct column from df_cn_ct
    # calculating lift coefficient
    cl_values = cn * np.cos(np.deg2rad(alpha)) - ct * np.sin(np.deg2rad(alpha))
    # calculating drag coefficient
    cd_values = cn * np.sin(np.deg2rad(alpha)) + ct * np.cos(np.deg2rad(alpha))
    # creating a Data Frame with column of lift coefficients and drag coefficients
    df_cl_cd = pd.DataFrame({'cl': cl_values, 'cd_stat': cd_values})
    
    # apply wind tunnel wall correction
    lambda_series = lambda_wall_corr['lambda'] #lambda_wall_corr is a Series
    df_cl_cd['cl'] = df_cl_cd['cl'] * (1 - 2 * lambda_series * (sigma_wall_corr + xi_wall_corr)-sigma_wall_corr)
    df_cl_cd['cd_stat'] = df_cl_cd['cd_stat'] * (1 - 2 * lambda_series * (sigma_wall_corr + xi_wall_corr))
    
    
    return df_cl_cd

def sort_rake_data(df_sync, num_columns):
    """
    prepares dataframe for cd calculation with ptot_rake_ , pstat_rake_ , ptot and pstat
    :param df_sync:                 synchronized data
    :param num columns:             number of measuring points of ptot_rake (usually 32)
    :return:df_sync_rake_sort       dataframe with necessary data for cd calculation
    """
    # extracts index of df_sync (=time) and generates new pandas dataframe df_rake_stat
    index = df_sync.index
    columns = [f'pstat_rake_{i+1}' for i in range(num_columns)]
    df_rake_stat = pd.DataFrame(index=index, columns=columns)
    
    # Fill four columns with measured pstat of wake rake
    df_rake_stat.iloc[:, 0] = df_sync['pstat_rake_1']
    df_rake_stat.iloc[:, 10] = df_sync['pstat_rake_2']
    df_rake_stat.iloc[:, 21] = df_sync['pstat_rake_3']
    df_rake_stat.iloc[:, 31] = df_sync['pstat_rake_4']
    
    # Interpolate the columns
    df_sync_rake_sort = df_rake_stat.astype(float).interpolate(axis=1)
    
    # Add the 'pstat' and 'ptot' columns from df_sync_stat_sort
    df_sync_rake_sort['pstat'] = df_sync_stat_sort['pstat'].values
    df_sync_rake_sort['ptot'] = df_sync_stat_sort['ptot'].values
    
    filtered_columns = [col for col in df_sync.columns if col.startswith('ptot_rake_')]

    # Insert filtered columns at the beginning of df_sync_rake_sort
    for col in reversed(filtered_columns):
        df_sync_rake_sort.insert(0, col, df_sync[col])

    return df_sync_rake_sort

def calc_rake_cd(df, lambda_wall_corr, sigma_wall_corr, xi_wall_corr):
    """
    Calculates drag coefficient cd based on wake rake measurements, provided by pandas dataframe df_sync_rake_sort
    calculation according Hinz eq. 2.12, applying wind tunnel correction according Althasus eq. 36
    :param df:              dataframe with synchronized data; columns: ptot_rake_i and pstat_rake_i (i=1...32; from wake rake) /
                            pstat and ptot from free flow (provided by pitot static unit)
    :return:df_cd_rake      dataframe with drag coefficient cd
    """
    # Create a list of column names for ptot_rake_i and pstat_rake_i
    ptot_cols = [f'ptot_rake_{i}' for i in range(1, 33)]
    pstat_cols = [f'pstat_rake_{i}' for i in range(1, 33)]
    
    # Verify columns exist in the DataFrame
    for col in  ['pstat', 'ptot'] + ptot_cols + pstat_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
    
    # Create a new dataframe with the deltas
    df_delta_numA = abs(df[ptot_cols].values - df[pstat_cols].values)
    df_delta_numB = abs(df[ptot_cols].values - df['pstat'].values[:, None])
    df_delta_denom = abs(df['ptot'].values[:, None] - df['pstat'].values[:, None])
    
    # Identify negative values and set them to NaN (negative square root!)
    df_delta_numA[df_delta_numA < 0] = np.nan
    df_delta_numB[df_delta_numB < 0] = np.nan
    df_delta_denom[df_delta_denom < 0] = np.nan
    
    # Perform the calculations
    result = np.sqrt(df_delta_numA / df_delta_denom) * (1 - np.sqrt(df_delta_numB / df_delta_denom))
    
    # Define trapezoidal rule integration
    def trapezoidal_rule(y, x):
        """
        Apply the trapezoidal rule for integration.
        :param y: array of function values at the sample points
        :param x: array of sample points
        :return: numerical integral
        """
        return np.trapz(y, x)
    
    # Assuming positions of the measurement points are equidistant
    x_rake = np.linspace(0, 1, 32)
    # integrating
    cd_rake = np.apply_along_axis(lambda y: trapezoidal_rule(y[~np.isnan(y)], x_rake[~np.isnan(y)]), 1, result) * 2
    
    # Create a DataFrame with the cd_rake results and the same index as the df input dataframe
    df_cd_rake = pd.DataFrame({'cd_rake': cd_rake}, index=df.index)
    
    # apply wind tunnel wall correction
    lambda_series = lambda_wall_corr['lambda'] #lambda_wall_corr is a Series
    df_cd_rake['cd_rake'] = df_cd_rake['cd_rake'] * (1 - 2 * lambda_series * (sigma_wall_corr + xi_wall_corr))
    
    return df_cd_rake

def plot(df, start_time, end_time, column_name_x, column_name_y):
    """
    generates plots of pandas DataFrame containing Time in row index and the specified column name in 
    parameter column_name between specified time interval
    :param df:                  pandas dataframe
    :param start_time:          time at which plot begins
    :param start_time:          time at which plot ends
    :param column_name_x:       this column name will be plotted on x axis (if time: enter 'time')
    :param column_name_y:       this column name will be plotted on y axis
    :return:1                   placeholder return value
    """
    # Filter the DataFrame based on the start and end times
    if start_time is not None:
        df = df[df.index >= start_time]
    if end_time is not None:
        df = df[df.index <= end_time]
        
    # Plotting with Matplotlib
    plt.figure(figsize=(12, 6))
    if column_name_x == 'time':
        plt.plot(df.index, df[column_name_y], label=column_name_y, color='blue')
    else:
        plt.plot(df[column_name_x], df[column_name_y], label=column_name_y, color='blue')
    
    # Calculate the mean value of the filtered DataFrame
    mean_value = df[column_name_y].mean()
    # Add a horizontal line for the mean value
    plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean Value: {mean_value:.2f}')
    
    # Adding titles and labels
    if column_name_x == 'time':
        plt.title(f'{column_name_y} vs Time')
        plt.xlabel('Time')
    else:
        plt.title(f'{column_name_y} vs {column_name_x}')
        plt.xlabel(f'{column_name_y}')
    
    plt.ylabel(column_name_y)
    plt.legend()
    
    # Display the plot
    plt.show()
    
    return 1

def plot_cl_cd_rake(df_cl_cd, df_cd_rake, start_time, end_time):
    """
    specialized plot method to generate plot of pandas DataFrames; one containing cl, the other cd_rake between specified time interval
    :param df_cl_cd:            pandas dataframe containing 'cl' column
    :param df_cd_rake:          pandas dataframe containing 'cd_rake' column
    :param start_time:          time at which plot begins
    :param start_time:          time at which plot ends
    :return:1                   placeholder return value
    """
    # Filter the DataFrame based on the start and end times for df_cl_cd
    if start_time is not None:
        df_cl_cd = df_cl_cd[df_cl_cd.index >= start_time]
    if end_time is not None:
        df_cl_cd = df_cl_cd[df_cl_cd.index <= end_time]
    # Filter the DataFrame based on the start and end times for df_cd_rake
    if start_time is not None:
        df_cd_rake = df_cd_rake[df_cd_rake.index >= start_time]
    if end_time is not None:
        df_cd_rake = df_cd_rake[df_cd_rake.index <= end_time]
        
    # Plotting with Matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(df_cl_cd['cl'], df_cd_rake['cd_rake'], label='cl', color='blue')
    
    # Adding titles and labels
    plt.title('cl vs cw_rake')
    plt.xlabel('cw_rake')
    plt.ylabel('cl')
    plt.legend()
    
    # Display the plot
    plt.show()
    
    return 1

def plot_cl_alpha(df_cl_cd, df_sync, start_time, end_time):
    """
    specialized plot method to generate plot of pandas DataFrames; one containing cl, the other alpha between specified time interval
    :param df_cl_cd:            pandas dataframe containing 'cl' column
    :param df_cd_rake:          pandas dataframe containing 'cd_rake' column
    :param start_time:          time at which plot begins
    :param start_time:          time at which plot ends
    :return:1                   placeholder return value
    """
    # Filter the DataFrame based on the start and end times for df_cl_cd
    if start_time is not None:
        df_cl_cd = df_cl_cd[df_cl_cd.index >= start_time]
    if end_time is not None:
        df_cl_cd = df_cl_cd[df_cl_cd.index <= end_time]
    # Filter the DataFrame based on the start and end times for df_cd_rake
    if start_time is not None:
        df_sync = df_sync[df_sync.index >= start_time]
    if end_time is not None:
        df_sync = df_sync[df_sync.index <= end_time]
        
    # Plotting with Matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(df_cl_cd['cl'], df_sync['Alpha'], label='cl', color='blue')
    
    # Adding titles and labels
    plt.title('cl vs alpha')
    plt.xlabel('alpha')
    plt.ylabel('cl')
    plt.legend()
    
    # Display the plot
    plt.show()
    
    return 1

def calc_wall_correction_coefficients(df_airfoil, df_cp):
    """
    calculate wall correction coefficients according to
    Abbott and van Doenhoff 1945: Theory of Wing Sections
    and
    Althaus 2003: Tunnel-Wall Corrections at the Laminar Wind Tunnel
    :param df_x_yt_cpt: pandas DataFrame with x, yt, and cp values of decambered airfoil
    :return:
    """

    # model - wall distances:
    d1 = 0.7
    d2 = 1.582
    
    # drop ptot and pstat columns and transpose df_airfoil 
    df_airfoil = df_airfoil.drop(columns=['pstat', 'ptot'])
    df_airfoil = df_airfoil.T
    df_airfoil = df_airfoil.drop(columns=['name', 'inop sensors', 'port', 'sensor unit'])
    df_airfoil = df_airfoil / 1000

    # calculate surface contour gradient dy_t/dx as finite difference scheme. first and last value are calculated
    # with forward and backward difference scheme, respectively and all other values with central difference
    # scheme
    dyt_dx = np.zeros(len(df_airfoil.index))
    dyt_dx[0] = (df_airfoil["y [mm]"].iloc[1] - df_airfoil["y [mm]"].iloc[0]) / \
                (df_airfoil["x [mm]"].iloc[1] - df_airfoil["x [mm]"].iloc[0])
    dyt_dx[-1] = (df_airfoil["y [mm]"].iloc[-1] - df_airfoil["y [mm]"].iloc[-2]) / \
                    (df_airfoil["x [mm]"].iloc[-1] - df_airfoil["x [mm]"].iloc[-2])
    dyt_dx[1:-1] = (df_airfoil["y [mm]"].iloc[2:].values - df_airfoil["y [mm]"].iloc[:-2].values) / \
                      (df_airfoil["x [mm]"].iloc[2:].values - df_airfoil["x [mm]"].iloc[:-2].values)
    # calculate v/V_inf
    v_V_inf = np.sqrt(abs(1 - df_cp.values))

    # calculate lambda (warning: Lambda of Althaus is erroneus, first y factor forgotten)
    lambda_wall_corr = integrate.simpson(y=16/np.pi*df_airfoil["y [mm]"].values*v_V_inf *np.sqrt(1 + dyt_dx**2), x=df_airfoil["x [mm]"].values)
    lambda_wall_corr = pd.DataFrame(lambda_wall_corr, columns=['lambda'])
    lambda_wall_corr.index = df_cp.index

    # calculate sigma
    sigma_wall_corr = np.pi**2 / 48 * (l_ref/1000)**2 * 1/2 * (1/(2*d1) + 1/(2*d2))**2

    # correction for model influence on static reference pressure
    # TODO: Re-calculate this using a panel method or with potential flow theory
    xi_wall_corr = -0.00335 * (l_ref/1000)**2
    
    return lambda_wall_corr, sigma_wall_corr, xi_wall_corr

 


    
    




if __name__ == '__main__':
    file_path_drive = '20230804-235819_drive.dat'
    file_path_AOA = '20230804-235818_AOA.dat'
    file_path_pstat_K02 = '20230804-235818_static_K02.dat'
    file_path_pstat_K03 = '20230804-235818_static_K03.dat'
    file_path_pstat_K04 = '20230804-235818_static_K04.dat'
    file_path_ptot_rake = '20230804-235818_ptot_rake.dat'
    file_path_pstat_rake = '20230804-235818_pstat_rake.dat'
    file_path_airfoil = 'Messpunkte_Demonstrator.xlsx'

    alphas = read_AOA_file(file_path_AOA)
    alpha_mean = calc_mean(alphas, start_time, end_time)
    pstat_K02 = read_DLR_pressure_scanner_file(file_path_pstat_K02, n_sens=32, t0=alphas["Time"].iloc[0])
    pstat_K03 = read_DLR_pressure_scanner_file(file_path_pstat_K03, n_sens=32, t0=alphas["Time"].iloc[0])
    pstat_K04 = read_DLR_pressure_scanner_file(file_path_pstat_K04, n_sens=32, t0=alphas["Time"].iloc[0])
    ptot_rake = read_DLR_pressure_scanner_file(file_path_ptot_rake, n_sens=32, t0=alphas["Time"].iloc[0])
    pstat_rake = read_DLR_pressure_scanner_file(file_path_pstat_rake, n_sens=5, t0=alphas["Time"].iloc[0])
    df_sync = synchronize_data([pstat_K02, pstat_K03, pstat_K04, ptot_rake,pstat_rake, alphas])
    df_airfoil = read_airfoil_geometry(file_path_airfoil, n_mp=97)
    df_sync_stat_sort = sort_sync_data(df_airfoil, df_sync)
    df_cp = calc_cp(df_sync_stat_sort)
    df_cn_ct = calc_cn_ct(df_cp, df_airfoil, t=499)
    lambda_wall_corr, sigma_wall_corr, xi_wall_corr = calc_wall_correction_coefficients(df_airfoil, df_cp)
    df_cl_cd = calc_cl_cd(df_cn_ct, df_sync)
    df_sync_rake_sort = sort_rake_data(df_sync, num_columns=32)
    df_cd_rake = calc_rake_cd(df_sync_rake_sort, lambda_wall_corr, sigma_wall_corr, xi_wall_corr)
    #plot(df_cl_cd, start_time, end_time, 'time', 'cl')
    #plot(df_cl_cd, start_time, end_time, 'time', 'cd')
    plot(df_cd_rake, start_time, end_time, 'time', 'cd_rake')
    #plot(df_cl_cd, start_time, end_time, 'cd_stat', 'cl')
    #plot_cl_cd_rake(df_cl_cd, df_cd_rake, start_time, end_time)
    #plot_cl_alpha(df_cl_cd, df_sync, start_time, end_time)
    
    

    
    print('done')




