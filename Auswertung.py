# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:21:04 2024

@author: Besitzer
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
import math
from itertools import chain
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import interpolate, integrate, optimize, stats
#sys.path.append("/put_airfoilwinggeometry_source_here/")
from airfoilwinggeometry.AirfoilPackage import AirfoilTools as at





def read_AOA_file(filename, t0):
    """
    Converts raw AOA data to pandas DataFrame
    :param filename:       File name
    :return: alphas             pandas DataFrame with AOA values
    """
    # Read the entire file into a pandas DataFrame
    df = pd.read_csv(filename, sep='\s+', header=None, names=['Date', 'Time', 'Position', 'Turn'])
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
    df = df[['Time', 'Alpha']]

    # Convert start time to milliseconds since it is easier to handle arithmetic operations
    start_time_ms = t0.timestamp() * 1000

    # Calculate the time difference in milliseconds from the first row
    time_diff_ms = df['Time'] - df['Time'].iloc[0]

    # Add this difference to the start time (in milliseconds) and convert back to datetime
    df['Time'] = pd.to_datetime(pd.Timestamp(start_time_ms, unit='ms') + time_diff_ms, unit='ms')

    return df

def read_GPS(filename):
    df = pd.read_csv(filename, header=None)
    # Apply the parsing function to each row
    parsed_data = df.apply(parse_gprmc_row, axis=1)

    # Extract the columns from the parsed data
    df_parsed = pd.DataFrame(parsed_data.tolist(), columns=['Time', 'Latitude', 'Longitude', 'U_GPS'])

    # Drop rows with any None values (if any invalid GPRMC sentences)
    df_parsed = df_parsed.dropna()

    return df_parsed
def parse_gprmc_row(row):
    parts = row.tolist()
    parts = [str(part) for part in parts]
    if len(parts) >= 13 and parts[0] == '$GPRMC' and parts[2] == 'A':
        try:
            time_str = parts[1]
            date_str = parts[9]
            latitude = float(parts[3][:2]) + float(parts[3][2:]) / 60.0
            if parts[4] == 'S':
                latitude = -latitude
            longitude = float(parts[5][:2]) + float(parts[5][2:]) / 60.0
            if parts[6] == 'W':
                longitude = -longitude
            gps_speed = float(parts[7]) * 1.852/3.6

            # Convert time and date to datetime
            datetime_str = date_str + time_str
            seconds, microseconds = datetime_str.split('.')
            microseconds = microseconds.ljust(6, '0')  # Pad to ensure 6 digits
            datetime_str = f"{seconds}.{microseconds}"
            datetime_format = '%d%m%y%H%M%S.%f'
            datetime_val = pd.to_datetime(datetime_str, format=datetime_format)

            return datetime_val, latitude, longitude, gps_speed
        except (ValueError, IndexError) as e:
            # Handle any parsing errors
            return None, None, None, None
    else:
        return None, None, None, None

def read_drive(filename, t0):
    """
    --> Reads drive data of wake rake (position and speed) into pandas DataFrame
    --> combines Date and Time to one pandas datetime column
    --> drive time = master time
    :param filename:       File name
    :return: df            pandas DataFrame with drive data
    """
    # list of column numbers in file to be read
    col_use = [0, 1, 2]
    # how columns are named
    col_name = ['Date', 'Time', 'Drive']

    # read file
    df = pd.read_csv(filename, sep="\s+", skiprows = 1, header=None, names=col_name, usecols=col_use,
                     on_bad_lines='skip', engine='python')

    # Combine Date and Time into a single pandas datetime column
    df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # drop date column (date column may generate problems when synchronizing data)
    df = df.drop(columns='Date')

    # Convert start time to milliseconds since it is easier to handle arithmetic operations
    start_time_ms = t0.timestamp() * 1000

    # Calculate the time difference in milliseconds from the first row
    time_diff_ms = df['Time'] - df['Time'].iloc[0]

    # Add this difference to the start time (in milliseconds) and convert back to datetime
    df['Time'] = pd.to_datetime(pd.Timestamp(start_time_ms, unit='ms') + time_diff_ms, unit='ms')

    return df

def calc_mean(df, start_time, end_time):
    """
    Calculates the mean value of every column in the specified time interval
    :param filename:            File name
    :param start_time:          begin of interval
    :param end_time:            time of first interval
    :return:                    pandas DataFrame with absolute time and pressures
    """
    # Select rows between start_time and end_time
    selected_values = df.loc[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
    
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

    # loading data into DataFrame. First line is skipped by default, because data is usually incorrect
    df = pd.read_csv(filename, sep="\s+", skiprows=1, header=None, on_bad_lines='skip')
    # if not as many columns a number of sensors (+2 for time and timedelta columns), then raise an error
    assert len(df.columns) == n_sens+2

    # drop timedelta column
    df = df.iloc[:, :-1]

    # assign column names
    df.columns = columns

    # drop lines with missing data
    df = df.dropna().reset_index(drop=True)

    # remove outliers
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    # Calculate the time difference in milliseconds from the first row
    time_diff_ms = df['Time'] - df['Time'].iloc[0]

    # Add this difference to the start time (in milliseconds) and convert back to datetime
    df['Time'] = pd.to_datetime(start_time_ms + time_diff_ms, unit='ms')

    return df

def synchronize_data(merge_dfs_list):
    """
    synchronizes and interpolates sensor data, given in pandas DataFrames with a timestamp
    :param merge_dfs_list:      list of pandas DataFrames containing sensor data. Must contain "Time" column in
                                datetime format
    :return: merged_df          merged dataframe with all sensor data, interpolated according time
    """

    # Merge the DataFrames using merge_asof
    merged_df = merge_dfs_list[0]
    for df in merge_dfs_list[1:]:
        merged_df = pd.merge_asof(merged_df, df, on='Time', tolerance=pd.Timedelta('1ms'), direction='nearest')

    # Set the index to 't_abs' to use time-based interpolation
    merged_df.set_index('Time', inplace=True)

    # Interpolate missing values using time-based interpolation
    merged_df = merged_df.interpolate(method='time')

    return merged_df

def read_airfoil_geometry(filename, c, foil_source, eta_flap, pickle_file=""):
    """
    --> searchs for pickle file in WD, if not found it creates a new pickle file
    --> generates pandas DataFrame with assignment of sensor unit + port to measuring point from Excel and renames
    --> adds 's' positions of the measuring points from Excel (line coordinate around the profile,
        starting at trailing edge)
    --> reads 'Kommentar' column of excel and drops sensors with status 'inop'
    --> calculates x and y position of static pressure points with airfoil coordinate file
    --> calculates x_n and y_n normal vector, tangential to airfoil surface of static pressure points
        with airfoil coordinate file

    :param filename:            file name of Excel eg. "Messpunkte Demonstrator.xlsx".
    :param c:                   airfoil chord length
    :param foil_source:         string, path of airfoil coordinate file
    :param eta_flap:            flap deflection angle
    :param pickle_file:         path to pickle file with airfoil information
    :return df_airfoil:         DataFrame with info described above
    """

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            df, eta_flap_read = pickle.load(file)

    if not os.path.exists(pickle_file) or eta_flap_read != eta_flap:
        # initialize airfoilTools object
        foil = at.Airfoil(foil_source)
        if eta_flap != 0.0:
            foil.flap(xFlap=0.8, yFlap=0, etaFlap=15)

        # Read Excel file
        df = pd.read_excel(filename, usecols="A:F", skiprows=1, skipfooter=1)# Read the Excel file
        df = df.dropna(subset=['Sensor unit K', 'Sensor port'])
        df = df.drop(df[df["Kommentar"] == "inop"].index).reset_index(drop=True)
        df = df.astype({'Messpunkt': 'int32', 'Sensor unit K': 'int32', 'Sensor port': 'int32'})

        df["s"] = df["Position [mm]"]/(c*1000)
        df["x"] = np.nan
        df["y"] = np.nan

        df["x_n"] = np.nan
        df["y_n"] = np.nan


        u_taps = np.zeros(len(df.index))

        for i, s in enumerate(df["s"]):
            res = optimize.root_scalar(at.s_curve, args=(foil.tck, s), x0=0, fprime=at.ds)
            u_taps[i] = res.root
            coords_tap = interpolate.splev(u_taps[i], foil.tck)
            df.loc[i, "x"] = coords_tap[0]
            df.loc[i, "y"] = coords_tap[1]
            n_tap = np.dot(at.tangent(u_taps[i], foil.tck)[0], np.array([[0, -1], [1, 0]]))
            df.loc[i, "x_n"] = n_tap[0]
            df.loc[i, "y_n"] = n_tap[1]

        if pickle_file != "":
            with open(pickle_file, 'wb') as file:
                pickle.dump([df, eta_flap], file)

    return df

def calc_U_CAS(df, prandtl_data):
    """
    --> picks static port of pstat and ptot of free stream, measured by prandtl probe from df
    --> assumes density of air according to International Standard Atmosphere (ISA)
    --> calculates calibrated airspeed with definition of total pressure

    :param df:                  pandas DataFrame with synchronized and interpolated measurement data
    :param prandtl_data:        dict with "unit name static", "i_sens_static", "unit name total" and
                                "i_sens_total".
                                This specifies the sensor units and the index of the sensors of the Prandtl
                                probe total
                                pressure sensor and the static pressure sensor
    :return:                    df with added 'U_CAS' column
    """

    colname_total = prandtl_data['unit name total'] + '_' + str(prandtl_data['i_sens_total'])
    colname_static = prandtl_data['unit name static'] + '_' + str(prandtl_data['i_sens_static'])
    ptot = df[colname_total]
    pstat = df[colname_static]

    # density of air according to International Standard Atmosphere (ISA)
    rho_ISA = 1.225

    df['U_CAS'] = np.sqrt(2 * (ptot - pstat) / rho_ISA)
    return df

def calc_wind(df):
    """
    --> calculates wind component in free stream direction

    :param df:          pandas DataFrame containing 'U_CAS' and 'U_GPS' column
    :return: df         pandas DataFrame with wind component column
    """

    df['Wind'] = df['U_CAS'] - df['U_GPS']

    return df

def calc_cp(df, prandtl_data, pressure_data_ident_strings):
    """
    calculates pressure coefficient for each static port on airfoil

    :param df:                          pandas DataFrame with synchronized and interpolated measurement data
    :param prandtl_data:                dict with "unit name static", "i_sens_static", "unit name total" and
                                        "i_sens_total".
                                        This specifies the sensor units and the index of the sensors of the Prandtl
                                        probe total
                                        pressure sensor and the static pressure sensor
    :param pressure_data_ident_strings: list of strings, which are contained in column names, which identify
                                        pressure sensor data
    :return: df                         pandas DataFrame with pressure coefficient in "static_K0X_Y" columns for
                                        every
                                        measuring point
    """
    # picks names of prandtl sensors
    colname_total = prandtl_data['unit name total'] + '_' + str(prandtl_data['i_sens_total'])
    colname_static = prandtl_data['unit name static'] + '_' + str(prandtl_data['i_sens_static'])
    # picks columns with prandtl data
    ptot = df[colname_total]
    pstat = df[colname_static]

    # column names of all pressure sensor data
    pressure_cols = []
    for string in pressure_data_ident_strings:
        pressure_cols += [col for col in df.columns if string in col]

    # apply definition of c_p
    df[pressure_cols] = df[pressure_cols].apply(lambda p_col: (p_col - pstat)/(ptot - pstat))

    df.replace([np.inf, -np.inf], 0., inplace=True)

    return df

def calc_cl_cm_cdp(df, df_airfoil, flap_pivots=[]):
    """
    calculates lift coefficient
    :param df:                  list of pandas DataFrames containing cp values
    :param df_airfoil:          list of pandas DataFrames containing columns of x_n and y_n normal vectors of measuring points
                                on airfoil
    :return:df                  merged dataframe with added cl coefficient column
    """

    # calculate tap normal vector components on airfoil surface projected to aerodynamic coordinate system
    n_proj_z = np.dot(df_airfoil[['x_n', 'y_n']].to_numpy(), np.array([-np.sin(np.deg2rad(df['Alpha'])),
                                             np.cos(np.deg2rad(df['Alpha']))])).T
    n_proj_x = np.dot(df_airfoil[['x_n', 'y_n']].to_numpy(), np.array([np.cos(np.deg2rad(df['Alpha'])),
                                             np.sin(np.deg2rad(df['Alpha']))])).T

    # assign tap index to sensor unit and sensor port
    sens_ident_cols = ["static_K0{0:d}_{1:d}".format(df_airfoil.loc[i, "Sensor unit K"], df_airfoil.loc[i,
                        "Sensor port"]) for i in df_airfoil.index]

    # calculate cl
    cp = df[sens_ident_cols].to_numpy()
    df["cl"] = -integrate.simpson(cp * n_proj_z, x=df_airfoil['s'])

    # calculate pressure drag
    df["cdp"] = -integrate.simpson(cp * n_proj_x, x=df_airfoil['s'])

    n_taps = df_airfoil[['x_n', 'y_n']].to_numpy()
    s_taps = df_airfoil['s']

    # calculate hinge moment
    r_ref = np.tile(np.array([0.25, 0]), [len(df_airfoil.index), 1]) - df_airfoil[['x', 'y']].to_numpy()
    df["cm"] = -integrate.simpson(cp * np.tile(np.cross(n_taps, r_ref), [len(df.index), 1]),
                                  x=s_taps)

    # calculate hinge moment of trailing edge flap
    n_flaps = len(flap_pivots)
    flap_pivots = np.array(flap_pivots)
    TE_flap = False
    LE_flap = False
    if n_flaps >= 1:
        TE_flap = True
        flap_pivot_TE = flap_pivots
    if n_flaps > 1:
        LE_flap = True
        flap_pivot_LE = flap_pivots[0, :]
        flap_pivot_TE = flap_pivots[1, :]
    if TE_flap:
        r_ref_F = df_airfoil[['x', 'y']].to_numpy() - np.tile(flap_pivot_TE, [len(df_airfoil.index), 1])
        mask = df_airfoil['x'].to_numpy() >= flap_pivot_TE[0]
        df["cmr_TE"] = integrate.simpson(cp[:, mask] * np.tile(np.cross(n_taps[mask], r_ref_F[mask, :]),
                                              [len(df.index), 1]), x=s_taps[mask])
    if LE_flap:
        r_ref_F = df_airfoil[['x', 'y']].to_numpy() - np.tile(flap_pivot_LE, [len(df_airfoil.index), 1])
        mask = df_airfoil['x'].to_numpy() <= flap_pivot_LE[0]
        df["cmr_LE"] = integrate.simpson(cp[:, mask] * np.tile(np.cross(n_taps[mask], r_ref_F[mask, :]),
                                              [len(df.index), 1]), x=s_taps[mask])

    """fig, ax = plt.subplots()
    ax.plot(df_airfoil["x"], df_airfoil["y"], "k-")
    ax.plot(df_airfoil["x"], -df[sens_ident_cols].iloc[15000], "b.-")
    plt.axis("equal")

    fig2, ax2 = plt.subplots()
    ax3 = ax2.twinx()
    ax3.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "Alpha"], "y-", label=r"$\alpha$")
    ax2.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "cl"], "k-", label="$c_l$")
    #ax2.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "cd"], "r-", label="$c_d$")
    ax2.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "cm"], "r-", label="$c_{m}$")
    ax2.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "cmr_LE"], "g-", label="$c_{m,r,LE}$")
    ax2.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "cmr_TE"], "b-", label="$c_{m,r,TE}$")

    ax2.grid()
    fig2.legend()

    fig4, ax4, = plt.subplots()
    ax4.plot(df_airfoil["x"], -cp[40000, :])
    
    fig5, ax5 = plt.subplots()
    ax5.plot(df["Longitude"], df["Latitude"], "k-")
    ax5.plot(df.loc[df["U_CAS"] > 25, "Longitude"], df.loc[df["U_CAS"] > 25, "Latitude"], "g-")
    """

    return df

def calc_cd(df, l_ref):
    """

    :param df:
    :return:
    """

    h_stat = 100
    h_tot = 93

    z_stat = np.linspace(-h_stat / 2, h_stat / 2, 5, endpoint=True)
    z_tot = np.linspace(-h_tot / 2, h_tot / 2, 32, endpoint=True)
    # it is assumed, that 0th sensor is defective (omit that value)
    z_tot = z_tot[1:]

    cp_stat_raw = df.filter(regex='^pstat_rake_').to_numpy()
    cp_stat_int = interpolate.interp1d(z_stat, cp_stat_raw, kind="linear", axis=1)

    cp_stat = cp_stat_int(z_tot)

    cp_tot = df.filter(regex='^ptot_rake_').to_numpy()
    # it is assumed, that 0th sensor is defective (omit that value)
    cp_tot = cp_tot[:, 1:]

    # Measurement of Proﬁle Drag by the Pitot-Traverse Method
    d_cd_jones = 2 * np.sqrt(np.abs((cp_tot - cp_stat))) * (1 - np.sqrt(np.abs(cp_tot)))

    # integrate integrand with simpson rule
    cd = integrate.simpson(d_cd_jones, z_tot) * 1 / (l_ref*1000)

    return cd

def apply_calibration_offset(filename, df):

    with open(filename, "rb") as file:
        calibr_data = pickle.load(file)

    l_ref = calibr_data[6]

    # flatten calibration data list, order like df pressure sensors
    pressure_calibr_data = calibr_data[2] + calibr_data[3] + calibr_data[4] + calibr_data[1] + calibr_data[0]
    # append zero calibration offsets for alpha, Lat/Lon, U_GPS and Drive
    pressure_calibr_data += [0]*(len(df.columns) - len(pressure_calibr_data))

    df_calibr_pressures = pd.DataFrame(data=[pressure_calibr_data], columns=df.columns)
    # repeat calibration data
    df_calibr_pressures = df_calibr_pressures.loc[df_calibr_pressures.index.repeat(len(df.index))]
    df_calibr_pressures.index = df.index

    # Apply calibration offsets
    df = df - df_calibr_pressures

    return df, l_ref

def apply_calibration_20sec(df):
    """
    uses first 20 seconds to calculate pressure sensor calibration offsets
    :param df:
    :return:
    """

    # select only pressures
    df_pres = df.iloc[:, :len(df.columns)-5]

    # Select the first 20 seconds of data
    first_20_seconds = df_pres[df_pres.index < df_pres.index[0] + pd.Timedelta(seconds=20)]

    # Calculate the mean for each sensor over the first 20 seconds
    mean_values = first_20_seconds.mean(axis=0)

    # Use these means to calculate the offsets for calibration
    offsets = mean_values - mean_values.mean()

    # Apply the calibration to the entire DataFrame
    df.iloc[:, :len(df.columns)-5] = df.iloc[:, :len(df.columns)-5] - offsets

    return df

def calc_wall_correction_coefficients(df_airfoil, df_cp, l_ref):
    """
    calculate wall correction coefficients according to
    Abbott and van Doenhoff 1945: Theory of Wing Sections
    and
    Althaus 2003: Tunnel-Wall Corrections at the Laminar Wind Tunnel
    :param df_airfoil:      pandas dataframe with x and y position of airfoil contour
    :param df_cp:           pandas dataframe with cp values
    :return:                wall correction coefficients
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
    lambda_wall_corr = integrate.simpson(
        y=16 / np.pi * df_airfoil["y [mm]"].values * v_V_inf * np.sqrt(1 + dyt_dx ** 2), x=df_airfoil["x [mm]"].values)
    lambda_wall_corr = pd.DataFrame(lambda_wall_corr, columns=['lambda'])
    lambda_wall_corr.index = df_cp.index

    # calculate sigma
    sigma_wall_corr = np.pi ** 2 / 48 * l_ref**2 * 1 / 2 * (1 / (2 * d1) + 1 / (2 * d2)) ** 2

    # correction for model influence on static reference pressure
    # TODO: Re-calculate this using a panel method or with potential flow theory
    xi_wall_corr = -0.00335 * l_ref**2

    return lambda_wall_corr, sigma_wall_corr, xi_wall_corr










def calc_cl_cd(df_cn_ct, df_sync):
    """
    Calculates lift and drag coefficient. Drag coefficient derived from static pressure ports on airfoil!
    Equations from Döller page 41, applying wind tunnel correction according Althasus eq. 36
    :param df_cn_ct:        list of pandas DataFrames containing "cn" and "ct" column
    :param df_sync:         list of pandas DataFrames containing column "Alpha"

    :return: df_cl_cd:      merged dataframe with lift and drag coefficient at certain times
    """

    alpha = df_sync.loc[:, "Alpha"] # extracting the synchronized alpha column from df_sync
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





    
    




if __name__ == '__main__':

    if os.getlogin() == 'joeac':
        WDIR = "C:/WDIR/MoProMa_Auswertung/"
    else:
        WDIR = "D:/Python_Codes/Workingdirectory_Auswertung"

    file_path_drive = os.path.join(WDIR,  '20230926-1713_drive.dat')
    file_path_AOA = os.path.join(WDIR,  '20230926-1713_AOA.dat')
    file_path_pstat_K02 = os.path.join(WDIR,  '20230926-1713_static_K02.dat')
    file_path_pstat_K03 = os.path.join(WDIR,  '20230926-1713_static_K03.dat')
    file_path_pstat_K04 = os.path.join(WDIR,  '20230926-1713_static_K04.dat')
    file_path_ptot_rake = os.path.join(WDIR,  '20230926-1713_ptot_rake.dat')
    file_path_pstat_rake = os.path.join(WDIR,  '20230926-1713_pstat_rake.dat')
    file_path_GPS = os.path.join(WDIR,  '20230926-1713_GPS.dat')
    file_path_airfoil = os.path.join(WDIR,  'Messpunkte Demonstrator.xlsx')
    pickle_path_airfoil = os.path.join(WDIR,  'Messpunkte Demonstrator.p')
    pickle_path_calibration = os.path.join(WDIR,  '20230926-171332_sensor_calibration_data.p')

    flap_pivots = np.array([[0.325, 0.0], [0.87, -0.004]])

    prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                    "unit name total": "static_K04", "i_sens_total": 32}

    start_time = "2023-08-04 21:58:19"
    end_time = "2023-08-04 21:58:49"


    if os.getlogin() == 'joeac':
        foil_coord_path = ("C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Data Brezn/"
                           "01_Aerodynamic Design/01_Airfoil Optimization/B203/0969/B203-0.dat")
    else:
        foil_coord_path= "D:/Python_Codes/Workingdirectory_Auswertung/B203-0.dat"

    os.chdir(WDIR)

    # read sensor data
    GPS = read_GPS(file_path_GPS)
    drive = read_drive(file_path_drive, t0=GPS["Time"].iloc[0])
    alphas = read_AOA_file(file_path_AOA, t0=GPS["Time"].iloc[0])
    #alpha_mean = calc_mean(alphas, start_time, end_time)
    pstat_K02 = read_DLR_pressure_scanner_file(file_path_pstat_K02, n_sens=32, t0=GPS["Time"].iloc[0])
    pstat_K03 = read_DLR_pressure_scanner_file(file_path_pstat_K03, n_sens=32, t0=GPS["Time"].iloc[0])
    pstat_K04 = read_DLR_pressure_scanner_file(file_path_pstat_K04, n_sens=32, t0=GPS["Time"].iloc[0])
    ptot_rake = read_DLR_pressure_scanner_file(file_path_ptot_rake, n_sens=32, t0=GPS["Time"].iloc[0])
    pstat_rake = read_DLR_pressure_scanner_file(file_path_pstat_rake, n_sens=5, t0=GPS["Time"].iloc[0])

    # synchronize sensor data
    df_sync = synchronize_data([pstat_K02, pstat_K03, pstat_K04, ptot_rake,pstat_rake, alphas, GPS, drive])

    # apply calibration offset from calibration file
    #df_sync, l_ref = apply_calibration_offset(pickle_path_calibration, df_sync)

    # apply calibration offset from first 20 seconds
    l_ref = 0.5
    df_sync = apply_calibration_20sec(df_sync)

    # read airfoil data
    df_airfoil = read_airfoil_geometry(file_path_airfoil, c=l_ref, foil_source=foil_coord_path, eta_flap=0.0,
                                       pickle_file=pickle_path_airfoil)

    # calculate airspeed from pitot static system
    df_sync = calc_U_CAS(df_sync, prandtl_data)

    # calculate wind component
    df_sync = calc_wind(df_sync)

    # calculate pressure coefficients
    df_sync = calc_cp(df_sync, prandtl_data, pressure_data_ident_strings=['stat', 'ptot'])

    # calculate lift coefficients
    cl = calc_cl_cm_cdp(df_sync, df_airfoil, flap_pivots)

    # calculate drag coefficients
    cd = calc_cd(df_sync, l_ref)

    # calculate drag coefficients
    #cd = calc_cd(df, n_sens=32)

    # calculate wall correction coefficients
    #lambda_wall_corr, sigma_wall_corr, xi_wall_corr = calc_wall_correction_coefficients(df_airfoil, df_cp)

    print("done")





    #df_cl_cd = calc_cl_cd(df_cn_ct, df_sync)
    #df_sync_rake_sort = sort_rake_data(df_sync, num_columns=32)
    #df_cd_rake = calc_rake_cd(df_sync_rake_sort, lambda_wall_corr, sigma_wall_corr, xi_wall_corr)
    #plot(df_cl_cd, start_time, end_time, 'time', 'cl')
    #plot(df_cl_cd, start_time, end_time, 'time', 'cd_stat')
    #plot(df_cd_rake, start_time, end_time, 'time', 'cd_rake')
    #plot(df_cl_cd, start_time, end_time, 'cd_stat', 'cl')
    #plot_cl_cd_rake(df_cl_cd, df_cd_rake, start_time, end_time)
    #plot_cl_alpha(df_cl_cd, df_sync, start_time, end_time)
    
    

    
    print('done')




