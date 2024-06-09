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
import matplotlib
from matplotlib.dates import DateFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import simps
from scipy import interpolate, integrate, optimize, stats
#sys.path.append("/put_airfoilwinggeometry_source_here/")
from airfoilwinggeometry.AirfoilPackage import AirfoilTools as at





def read_AOA_file(filename, sigma_wall, t0):
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

    # apply wind tunnel wall corrections
    df.loc[:, "Alpha"] = df["Alpha"] * (1 + sigma_wall)

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
    """
    processes GPS data in gprmc format
    :param row:
    :return:
    """
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
    col_use = [0, 1, 2, 3]
    # how columns are named
    col_name = ['Date', 'Time', 'Rake Position', 'Rake Speed']

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

    # initialize airfoilTools object
    foil = at.Airfoil(foil_source)

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            df, eta_flap_read = pickle.load(file)

    if not os.path.exists(pickle_file) or eta_flap_read != eta_flap:

        if eta_flap != 0.0:
            foil.flap(xFlap=0.8, yFlap=0, etaFlap=15)

        # Read Excel file
        df = pd.read_excel(filename, usecols="A:F", skiprows=1, skipfooter=1)# Read the Excel file
        df = df.dropna(subset=['Sensor unit K', 'Sensor port'])
        df = df.drop(df[df["Kommentar"] == "inop"].index).reset_index(drop=True)
        df = df.astype({'Messpunkt': 'int32', 'Sensor unit K': 'int32', 'Sensor port': 'int32'})

        # append virtual trailing edge pressure taps (pressure is mean between last sensor at to and bottom side)
        df_virt_top = pd.DataFrame([[np.nan, "virtual_top", 0, -1, -1, np.nan]], columns=df.columns)
        # virtual bottom trailing edge tap: s value must be calculated
        df_virt_bot = pd.DataFrame([[np.nan, "virtual_bot", at.s_curve(foil.u[-1], foil.tck)*c*1000, -1, -1, np.nan]],
                                   columns=df.columns)
        df = pd.concat([df_virt_top, df, df_virt_bot]).reset_index(drop=True)

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

    return df, foil
def calc_airspeed_wind(df, prandtl_data, T, l_ref):
    """
    --> calculates wind component in free stream direction

    :param df:          pandas DataFrame containing 'U_CAS' and 'U_GPS' column
    :return: df         pandas DataFrame with wind component column
    """


    colname_total = prandtl_data['unit name total'] + '_' + str(prandtl_data['i_sens_total'])
    colname_static = prandtl_data['unit name static'] + '_' + str(prandtl_data['i_sens_static'])
    ptot = df[colname_total]
    pstat = df[colname_static]

    # density of air according to International Standard Atmosphere (ISA)
    rho_ISA = 1.225
    R_s = 287.0500676
    # calculate derived variables (dynamic viscosity). Has to be calculated online if we chose to add a temp sensor
    # Formula from https://calculator.academy/viscosity-of-air-calculator/
    mu = (1.458E-6 * T ** (3 / 2)) / (T + 110.4)

    df['U_CAS'] = np.sqrt(2 * (ptot - pstat) / rho_ISA)

    # calculate air speeds
    rho = pstat / (R_s * T)
    df['U_TAS'] = np.sqrt(np.abs(2 * (ptot - pstat) / rho))
    df['Re'] = df['U_TAS'] * l_ref * rho / mu

    # calculating wind component in free stream direction
    df['wind_component'] = df['U_TAS'] - df['U_GPS']

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
def calc_cl_cm_cdp(df, df_airfoil, at_airfoil, flap_pivots=[], lambda_wall=0., sigma_wall=0., xi_wall=0.):
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
    sens_ident_cols = ["static_K0{0:d}_{1:d}".format(df_airfoil.loc[i, "Sensor unit K"],
                                                     df_airfoil.loc[i, "Sensor port"]) for i in df_airfoil.index[1:-1]]
    # calculate virtual pressure coefficient
    df["static_virtual_top"] = df["static_virtual_bot"] = (df[sens_ident_cols[0]] + df[sens_ident_cols[-1]])/2
    # re-arrange columns
    cols = df.columns.to_list()
    cols = cols[:3*32] + cols[-2:] + cols[3*32:-2]
    df = df[cols].copy()
    sens_ident_cols = ["static_virtual_top"] + sens_ident_cols + ["static_virtual_bot"]


    # calculate cl
    cp = df[sens_ident_cols].to_numpy()
    df.loc[:, "cl"] = -integrate.simpson(cp * n_proj_z, x=df_airfoil['s'])

    # calculate pressure drag
    df.loc[:, "cdp"] = -integrate.simpson(cp * n_proj_x, x=df_airfoil['s'])

    n_taps = df_airfoil[['x_n', 'y_n']].to_numpy()
    s_taps = df_airfoil['s']

    # calculate hinge moment
    r_ref = np.tile(np.array([0.25, 0]), [len(df_airfoil.index), 1]) - df_airfoil[['x', 'y']].to_numpy()
    df.loc[:, "cm"] = -integrate.simpson(cp * np.tile(np.cross(n_taps, r_ref), [len(df.index), 1]),
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
        df.loc[:, "cmr_TE"] = integrate.simpson(cp[:, mask] * np.tile(np.cross(n_taps[mask], r_ref_F[mask, :]),
                                              [len(df.index), 1]), x=s_taps[mask])
    if LE_flap:
        r_ref_F = df_airfoil[['x', 'y']].to_numpy() - np.tile(flap_pivot_LE, [len(df_airfoil.index), 1])
        mask = df_airfoil['x'].to_numpy() <= flap_pivot_LE[0]
        df.loc[:, "cmr_LE"] = integrate.simpson(cp[:, mask] * np.tile(np.cross(n_taps[mask], r_ref_F[mask, :]),
                                              [len(df.index), 1]), x=s_taps[mask])
        # apply wind tunnel wall corrections
        df.loc[:, "cl"] = df["cl"] * (1 - 2 * lambda_wall * (sigma_wall + xi_wall) - sigma_wall)

        df.loc[:, "cm"] = df["cm"] * (1 - 2 * lambda_wall * (sigma_wall + xi_wall))

    # finally apply wall correction to cp's (after calculation of lift and moment coefficients.
    # Otherwise, correction would be applied twice
    df.loc[:, sens_ident_cols] = (1 - 2 * lambda_wall * (sigma_wall + xi_wall) - sigma_wall) * df[sens_ident_cols]

    return df, sens_ident_cols, cp
def calc_cd(df, l_ref, lambda_wall, sigma_wall, xi_wall):
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

    # apply wind tunnel wall corrections
    cd = cd * (1 - 2 * lambda_wall * (sigma_wall + xi_wall))

    df["cd"] = cd

    return df

def apply_calibration_offset(filename, df):

    with open(filename, "rb") as file:
        calibr_data = pickle.load(file)

    l_ref = calibr_data[6]

    # flatten calibration data list, order like df pressure sensors
    pressure_calibr_data = calibr_data[2] + calibr_data[3] + calibr_data[4] + calibr_data[1] + calibr_data[0]
    # append zero calibration offsets for alpha, Lat/Lon, U_GPS and Rake Position
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
    df_pres = df.iloc[:, :len(df.columns)-6]

    # Select the first 20 seconds of data
    first_20_seconds = df_pres[df_pres.index < df_pres.index[0] + pd.Timedelta(seconds=20)]

    # Calculate the mean for each sensor over the first 20 seconds
    mean_values = first_20_seconds.mean(axis=0)

    # Use these means to calculate the offsets for calibration
    offsets = mean_values - mean_values.mean()

    # Apply the calibration to the entire DataFrame
    df.iloc[:, :len(df.columns)-6] = df.iloc[:, :len(df.columns)-6] - offsets

    return df
def calc_wall_correction_coefficients(df_airfoil, filepath, l_ref):
    """
    calculate wall correction coefficients according to
    Abbott and van Doenhoff 1945: Theory of Wing Sections
    and
    Althaus 2003: Tunnel-Wall Corrections at the Laminar Wind Tunnel
    :param df_airfoil:      pandas dataframe with x and y position of airfoil contour
    :param df_cp:           pandas dataframe with cp values
    :return:                wall correction coefficients
    """

    #read symmetrical airfoil data
    column_names = ['x', 'y', 'cp']
    df_wall_correction_cp = pd.read_csv(filepath, names=column_names, skiprows=3, delim_whitespace=True)

    # model - wall distances:
    d1 = 0.7
    d2 = 1.582

    # cut off bottom airfoil side
    df_wall_correction_cp = df_wall_correction_cp.iloc[:np.argmin(df_wall_correction_cp["x"]) + 1, :]
    # and flip it
    df_wall_correction_cp = df_wall_correction_cp.iloc[::-1, :].reset_index(drop=True)

    # calculate surface contour gradient dy_t/dx as finite difference scheme. first and last value are calculated
    # with forward and backward difference scheme, respectively and all other values with central difference
    # scheme
    dyt_dx = np.gradient(df_wall_correction_cp["y"])/np.gradient(df_wall_correction_cp["x"])

    # calculate v/V_inf
    v_V_inf = np.sqrt(1 - df_wall_correction_cp["cp"].values)

    # calculate lambda (warning: Lambda of Althaus is erroneus, first y factor forgotten)
    lambda_wall_corr = integrate.simpson(y=16 / np.pi * df_wall_correction_cp["y"].values * v_V_inf *
                                                np.sqrt(1 + dyt_dx ** 2), x=df_wall_correction_cp["x"].values)

    # calculate sigma
    sigma_wall_corr = np.pi ** 2 / 48 * l_ref**2 * 1 / 2 * (1 / (2 * d1) + 1 / (2 * d2)) ** 2

    # correction for model influence on static reference pressure
    # TODO: Re-calculate this using a panel method or with potential flow theory
    xi_wall_corr = -0.00335 * l_ref**2

    return lambda_wall_corr, sigma_wall_corr, xi_wall_corr
def plot_specify_section(df, cp):
    """

    :param df_sync:
    :return:
    """



    plt.close('all')

    # plot U_CAS over time
    fig, ax, = plt.subplots()
    ax.plot(df["U_CAS"])
    ax.set_xlabel("$Time$")
    ax.set_ylabel("$U_{CAS} [m/s]$")
    ax.set_title("$U_{CAS}$ vs. Time")
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    # plot alpha, cl, cm, cmr over time
    fig3, ax3 = plt.subplots()
    ax4 = ax3.twinx()
    ax4.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "Alpha"], "y-", label=r"$\alpha$")
    ax3.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "cl"], "k-", label="$c_l$")
    ax3.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "cm"], "r-", label="$c_{m}$")
    ax3.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "cmr_LE"], "g-", label="$c_{m,r,LE}$")
    ax3.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "cmr_TE"], "b-", label="$c_{m,r,TE}$")
    ax3.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax3.grid()
    fig3.legend()

    """
    # plot path of car
    fig5, ax5 = plt.subplots()
    ax5.plot(df["Longitude"], df["Latitude"], "k-")
    ax5.plot(df.loc[df["U_CAS"] > 25, "Longitude"], df.loc[df["U_CAS"] > 25, "Latitude"], "g-")
    """

    # plot c_d, rake position and rake speed over time
    fig6, ax6 = plt.subplots()
    ax7 = ax6.twinx()
    ax6.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "cd"], "b-", label="$c_d$")
    ax7.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "Rake Position"], "r-", label="rake position")
    ax7.plot(df.loc[df["U_CAS"] > 25].index, df.loc[df["U_CAS"] > 25, "Rake Speed"], "g-", label="rake speed")
    ax6.set_xlabel("$Time$")
    ax7.set_xlabel("Rake Position / Speed")
    ax6.set_ylabel("$c_d$")
    ax7.set_ylabel("$Rake Position [mm]$")
    ax6.set_title("$c_d$ vs. Time")
    ax6.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    fig6.legend()
    ax6.grid()



    plt.show()

    return 1
def plot_operating_points(df, df_airfoil, at_airfoil, sens_ident_cols, t):
    """
    plots cp(x) and wake depression (x) at certain operating points (alpha, Re and beta)
    :param df:      pandas dataframe with index time and data to be plotted
    :param t:       index number of operating point (=time)
    :return:
    """
    h_stat = 100
    h_tot = 93

    # plot cp(x)
    fig, ax = plt.subplots()
    ax_cp = ax.twinx()
    ax.plot(at_airfoil.coords[:, 0], at_airfoil.coords[:, 1], "k-")
    ax.plot(df_airfoil["x"], df_airfoil["y"], "k.")
    ax_cp.plot(df_airfoil["x"], df[sens_ident_cols].iloc[t], "r.-")
    ylim_u, ylim_l = ax_cp.get_ylim()
    ax_cp.set_ylim([ylim_l, ylim_u])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax_cp.set_ylabel("$c_p$")
    ax_cp.grid()
    ax.axis("equal")

    # plot wake depression(x)
    # extract total pressure of wake rake from dataframe
    cols = df.columns.to_list()
    cols = df.filter(regex='^ptot')
    # it is assumed, that 0th sensor is defective (omit that value)
    cols = cols.iloc[:,1:]

    # positions of total pressure sensors of wake rake
    z_tot = np.linspace(-h_tot / 2, h_tot / 2, 32, endpoint=True)
    # bring it to similar dimensions of airfoil
    z_tot = z_tot / 100;
    # it is assumed, that 0th sensor is defective (omit that value)
    z_tot = z_tot[1:]

    # for better appearance, move airfoil to wake depression
    #df_airfoil_y_corr = df_airfoil["y"] + 0.2
    df_airfoil_y_corr = at_airfoil.coords[:, 1]+0.16

    fig, ax = plt.subplots()
    ax_cp = ax.twiny()
    ax.plot(at_airfoil.coords[:, 0], -df_airfoil_y_corr, "k-")
    ax_cp.plot(cols.iloc[t], z_tot, "r.-")
    ylim_u, ylim_l = ax_cp.get_ylim()
    ax_cp.set_ylim([ylim_l, ylim_u])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax_cp.set_xlabel("$c_p$")
    ax.set_title("Wake Depression")
    ax_cp.grid()
    ax.axis("equal")


    plt.show()

    return 1
def plot_3D(df):
    """

    :param df:
    :return:
    """


    # Create a new figure for the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract data for the plot
    x = df.index
    y = df['U_CAS']
    z = df['Rake Position']

    # Convert the datetime index to a numerical format for plotting
    x_num = x.map(pd.Timestamp.toordinal)

    # Create the 3D scatter plot
    ax.scatter(x_num, y, z, c='r', marker='o')

    # Set labels
    ax.set_xlabel('Time')
    ax.set_ylabel('c_d')
    ax.set_zlabel('Rake Position')

    # Convert the numerical x-axis back to dates for better readability
    ax.set_xticklabels(x.strftime("%H:%M"))

    # Show the plot
    plt.show()

    return 1
def calc_mean(df, alpha, Re):
    """
    calculates mean values of AOA, lift-, drag- and moment coefficients for a given alpha (automativally given from
    calc_means when called) and an entered Reynoldsnumber
    :param df:          pandas dataframe with column names: alpha, cl, cd, cm (given from plot polars function)
    :param alpha:       automativally given from calc_means when called
    :param Re:          desired Reynoldsnumber for the test (needs to be typed in function call)
    :return:
    """
    # define Intervalls (might be adapted)
    delta_alpha = 0.2
    min_alpha = alpha - delta_alpha
    max_alpha = alpha + delta_alpha
    delta_Re = 0.2e6
    min_Re = Re - delta_Re
    max_Re = Re + delta_Re

    # conditions to be representive values
    condition = ((df["Alpha"] > min_alpha) &
                 (df["Alpha"] < max_alpha) &
                 (df["Re"] > min_Re) &
                 (df["Re"] < max_Re) &
                 (df["Rake Speed"] != 0))

    # pick values which fulfill the condition
    col_alpha = df.loc[condition, "Alpha"]
    col_cl = df.loc[condition, "cl"]
    col_cd = df.loc[condition, "cd"]
    col_cm = df.loc[condition, "cm"]

    # calculate mean values
    mean_alpha = col_alpha.mean()
    mean_cl = col_cl.mean()
    mean_cd = col_cd.mean()
    mean_cm = col_cm.mean()

    return mean_alpha, mean_cl, mean_cd, mean_cm
def prepare_polar_df(df, Re, alpha_range=range(1, 18)):
    """
    iterates over alpha [1,17] deg and calculates to each alpha the mean values of cl, cd and cm; if alpha and Re
    criteria are not fulfilled, moves on to next alpha value
    :param df:                  pandas dataframe with all data to be plotted
    :param alpha_range:         AOA interval for polar
    :param Re:                  desired Reynoldsnumber for polar
    :return: df_polars            df with polar values ready to be plotted
    """
    # create a new dataframe with specified column names
    df_polars = pd.DataFrame(columns=["alpha", "cl", "cd", "cm"])

    # iterates over alpha and calls calc_mean function
    for alpha in alpha_range:
        mean_alpha, mean_cl, mean_cd, mean_cm = calc_mean(df, alpha, Re)
        if not pd.isna(mean_cl):  # just added if mean value can be calculated
            new_row = pd.DataFrame({"alpha": [alpha], "cl": [mean_cl], "cd": [mean_cd], "cm": [mean_cm]})
            df_polars = pd.concat([df_polars, new_row], ignore_index=True)

    return df_polars
def plot_polars(df):
    """

    :param df_polars:
    :return:
    """
    # just for testing
    #********************************************************************************
    data = {
        'alpha': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        'cl': [0.01, 0.1, 0.15, 0.19, 0.23, 0.27, 0.28, 0.4, 0.45, 0.49,
               0.6, 0.7, 0.9, 1.3, 1.4, 1.2, 0.99],
        'cd': [0.01, 0.1, 0.15, 0.19, 0.23, 0.27, 0.28, 0.4, 0.45, 0.49,
               0.6, 0.7, 0.9, 1.3, 1.4, 1.2, 0.99],
        'cm': [0.01, 0.1, 0.15, 0.19, 0.23, 0.27, 0.28, 0.4, 0.45, 0.49,
               0.6, 0.7, 0.9, 1.3, 1.4, 1.2, 0.99]
    }
    df = pd.DataFrame(data)
    #********************************************************************************


    # plot cl(alpha)
    fig, ax = plt.subplots()
    ax.plot(df["alpha"], df["cl"], "k.", linestyle='-')
    ax.set_xlabel("$alpha$")
    ax.set_ylabel("$c_l$")
    ax.set_title("$c_l$ vs. alpha")
    ax.grid()

    # plot cl(cd)
    fig2, ax2 = plt.subplots()
    ax2.plot(df["cd"], df["cl"], "k.", linestyle='-')
    ax2.set_xlabel("$c_d$")
    ax2.set_ylabel("$c_l$")
    ax2.set_title("$c_l$ vs. $c_d$")
    ax2.grid()

    # plot cl(cm)
    fig3, ax3 = plt.subplots()
    ax3.plot(df["cm"], df["cl"], "k.", linestyle='-')
    ax3.set_xlabel("$c_m$")
    ax3.set_ylabel("$c_l$")
    ax3.set_title("$c_l$ vs. $c_m$")
    ax3.grid()

    # plot cm(alpha)
    fig4, ax4 = plt.subplots()
    ax4.plot(df["alpha"], df["cm"], "k.", linestyle='-')
    ax4.set_xlabel("$alpha$")
    ax4.set_ylabel("$c_m$")
    ax4.set_title("$c_m$ vs. alpha")
    ax4.grid()

    plt.show()

    return 1




if os.getlogin() == 'joeac':
    WDIR = "C:/WDIR/MoProMa_Auswertung/"
else:
    WDIR = "D:/Python_Codes/Workingdirectory_Auswertung"

file_path_drive = os.path.join(WDIR, '20230926-1713_drive.dat')
file_path_AOA = os.path.join(WDIR, '20230926-1713_AOA.dat')
file_path_pstat_K02 = os.path.join(WDIR, '20230926-1713_static_K02.dat')
file_path_pstat_K03 = os.path.join(WDIR, '20230926-1713_static_K03.dat')
file_path_pstat_K04 = os.path.join(WDIR, '20230926-1713_static_K04.dat')
file_path_ptot_rake = os.path.join(WDIR, '20230926-1713_ptot_rake.dat')
file_path_pstat_rake = os.path.join(WDIR, '20230926-1713_pstat_rake.dat')
file_path_GPS = os.path.join(WDIR, '20230926-1713_GPS.dat')
file_path_airfoil = os.path.join(WDIR, 'Messpunkte Demonstrator.xlsx')
pickle_path_airfoil = os.path.join(WDIR, 'Messpunkte Demonstrator.p')
pickle_path_calibration = os.path.join(WDIR, '20230926-171332_sensor_calibration_data.p')
cp_path_wall_correction = os.path.join(WDIR, 'B200-0_reinitialized.cp')

flap_pivots = np.array([[0.325, 0.0], [0.87, -0.004]])

prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                "unit name total": "static_K04", "i_sens_total": 32}

if os.getlogin() == 'joeac':
    foil_coord_path = ("C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Data Brezn/"
                       "01_Aerodynamic Design/01_Airfoil Optimization/B203/0969/B203-0.dat")
else:
    foil_coord_path = "D:/Python_Codes/Workingdirectory_Auswertung/B203-0.dat"

os.chdir(WDIR)



# read airfoil data
l_ref = 0.5
df_airfoil, airfoil = read_airfoil_geometry(file_path_airfoil, c=l_ref, foil_source=foil_coord_path, eta_flap=0.0,
                                            pickle_file=pickle_path_airfoil)

# calculate wall correction coefficients
lambda_wall, sigma_wall, xi_wall = calc_wall_correction_coefficients(df_airfoil, cp_path_wall_correction, l_ref)

# read sensor data
GPS = read_GPS(file_path_GPS)
drive = read_drive(file_path_drive, t0=GPS["Time"].iloc[0])
alphas = read_AOA_file(file_path_AOA, sigma_wall, t0=GPS["Time"].iloc[0])
pstat_K02 = read_DLR_pressure_scanner_file(file_path_pstat_K02, n_sens=32, t0=GPS["Time"].iloc[0])
pstat_K03 = read_DLR_pressure_scanner_file(file_path_pstat_K03, n_sens=32, t0=GPS["Time"].iloc[0])
pstat_K04 = read_DLR_pressure_scanner_file(file_path_pstat_K04, n_sens=32, t0=GPS["Time"].iloc[0])
ptot_rake = read_DLR_pressure_scanner_file(file_path_ptot_rake, n_sens=32, t0=GPS["Time"].iloc[0])
pstat_rake = read_DLR_pressure_scanner_file(file_path_pstat_rake, n_sens=5, t0=GPS["Time"].iloc[0])

# synchronize sensor data
df_sync = synchronize_data([pstat_K02, pstat_K03, pstat_K04, ptot_rake, pstat_rake, alphas, GPS, drive])

# apply calibration offset from calibration file
#df_sync, l_ref = apply_calibration_offset(pickle_path_calibration, df_sync)

# apply calibration offset from first 20 seconds
T_air = 288
df_sync = apply_calibration_20sec(df_sync)

# calculate wind component
df_sync = calc_airspeed_wind(df_sync, prandtl_data, T_air, l_ref)

# calculate pressure coefficients
df_sync = calc_cp(df_sync, prandtl_data, pressure_data_ident_strings=['stat', 'ptot'])

# calculate lift coefficients
df_sync, sens_ident_cols, cp = calc_cl_cm_cdp(df_sync, df_airfoil, airfoil, flap_pivots, lambda_wall, sigma_wall, xi_wall)

# calculate drag coefficients
df_sync = calc_cd(df_sync, l_ref, lambda_wall, sigma_wall, xi_wall)

# visualisation
plot_specify_section(df_sync, cp)
plot_3D(df_sync)
plot_operating_points(df_sync, df_airfoil, airfoil, sens_ident_cols, t=40000)

df_polars = prepare_polar_df(df_sync, Re=1e6)
plot_polars(df_polars)

print("done")

