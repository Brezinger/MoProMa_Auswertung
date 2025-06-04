# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:21:04 2024

@author: Besitzer
"""
import copy
import sys
import os
import re
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import itertools
import pynmea2
from pyproj import Transformer

from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True

from scipy.signal import  savgol_filter
from scipy import interpolate, integrate, optimize, stats
if os.getlogin() == 'joeac':
    sys.path.append("C:/git/airfoilwinggeometry")
else:
    sys.path.append("D:/Python_Codes/Uebung1/modules/airfoilwinggeometry")
from airfoilwinggeometry.AirfoilPackage import AirfoilTools as at

def asymmetric_trim_mean(row, lower_frac, upper_frac):
    "Calculates trimmed mean values for reference total pressure calculation"
    sorted_row = sorted(row)
    n = len(sorted_row)
    lower_idx = int(n * lower_frac)
    upper_idx = int(n * (1 - upper_frac))
    trimmed_values = sorted_row[lower_idx:upper_idx]
    return sum(trimmed_values) / len(trimmed_values)

def trimmed_median(row, lower_frac=0.5, upper_frac=0.05):
    sorted_row = np.sort(row)
    n = len(sorted_row)
    lower_idx = int(np.floor(n * lower_frac))
    upper_idx = int(np.ceil(n * (1 - upper_frac)))
    trimmed_values = sorted_row[lower_idx:upper_idx]
    return np.median(trimmed_values)

def read_AOA_file(filename, sigma_wall, t0, alpha_sens_offset=214.73876953125):
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
    abs_sensor_pos_deg = - df['Position'] / 2**14 * 360 - df['Turn'] * 360 + alpha_sens_offset
    # Compute the gear ratio and alpha
    gear_ratio = 60 / (306 * 2)
    df['alpha'] = abs_sensor_pos_deg * gear_ratio

    # Combine Date and Time into a single pandas datetime column
    df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Select only the relevant columns
    df = df[['Time', 'alpha']]

    # Convert start time to milliseconds since it is easier to handle arithmetic operations
    start_time_ms = t0.timestamp() * 1000

    # Calculate the time difference in milliseconds from the first row
    time_diff_ms = df['Time'] - df['Time'].iloc[0]

    # Calculate time difference between t0 (GPS data) and AOA data (from local computer time)
    delta_t_GPS_PC = t0.astimezone(timezone.utc) - df['Time'].iloc[0].tz_localize(timezone.utc)

    # Add this difference to the start time (in milliseconds) and convert back to datetime
    df['Time'] = pd.to_datetime(pd.Timestamp(start_time_ms, unit='ms') + time_diff_ms, unit='ms')
    df['Time'] = df['Time'].dt.tz_localize('UTC')

    # apply wind tunnel wall corrections
    df.loc[:, "alpha"] = df["alpha"] * (1 + sigma_wall)

    return df, delta_t_GPS_PC

def read_GPS(filename):

    with open(filename) as file:
        lns = file.readlines()

    df = pd.DataFrame(lns)
    # Apply the parsing function to each row
    parsed_data = df.apply(parse_gprmc_row, axis=1)

    # Extract the columns from the parsed data
    df_parsed = pd.DataFrame(parsed_data.tolist(), columns=['Time', 'Latitude', 'Longitude', 'U_GPS'])

    # Drop rows with any None values (if any invalid GPRMC sentences)
    df_parsed = df_parsed.dropna()



    return df_parsed

def parse_gprmc_row(line):
    if "$GPRMC" in line.values[0] or "$GNRMC" in line.values[0]:
        data = pynmea2.parse(line.values[0])
        if data.is_valid:
            try:
                # Get the time
                timestamp = data.datetime
                # Get the speed in knots
                speed_knots = data.spd_over_grnd
                lat = data.latitude
                lon = data.longitude
                # Convert the speed to km/h (1 knot = 1.852 km/h)
                speed_ms = speed_knots * 1.852/3.6
                return timestamp, lat, lon, speed_ms
            except TypeError:
                return None, None, None, None
        else:
            return None, None, None, None
    else:
        return None, None, None, None

def read_drive(filename, t0, delta_t, sync_method="delta_t"):
    """
    --> Reads drive data of wake rake (position and speed) into pandas DataFrame
    --> combines Date and Time to one pandas datetime column
    --> drive time = master time
    :param filename:      File name
    :param t0:            Start time of first timestamp for synchronization
    :param delta_t        Time offset between GPS time and computer time of measurement laptop
    :param sync_method:   Synchronization method to use. Either t0 to use first timestamp and set times equal (better,
                          when first timestamp was recorded), or delta_t (better, when first timestamp was not recorded)
    :return: df           pandas DataFrame with drive data
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
    if len(df.index) != 0:
        if df['Time'].iloc[0].tzinfo is None or df['Time'].iloc[0].tzinfo.utcoffset(df['Time'].iloc[0]) is None:
            df['Time'] = (
                df['Time']
                .dt.tz_localize('Europe/Vienna')  # Set the correct local timezone
                .dt.tz_convert('UTC')  # Convert to UTC
            )

        # drop date column (date column may generate problems when synchronizing data)
        df = df.drop(columns='Date')

        if sync_method == "t0":
            # Convert start time to milliseconds since it is easier to handle arithmetic operations
            start_time_ms = t0.timestamp() * 1000

            # Calculate the time difference in milliseconds from the first row
            if len(df.index) > 0:
                time_diff_ms = df['Time'] - df['Time'].iloc[0]

                # Add this difference to the start time (in milliseconds) and convert back to datetime
                df['Time'] = pd.to_datetime(pd.Timestamp(start_time_ms, unit='ms') + time_diff_ms, unit='ms')
        elif sync_method == "delta_t":
            df['Time'] = df['Time'] + delta_t
        else:
            raise ValueError("wrong sync method")

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
    df = pd.read_csv(filename, sep="\s+", skiprows=1, header=None, on_bad_lines='skip').dropna(axis=1, how='all')
    # if not as many columns a number of sensors (+2 for time and timedelta columns), then raise an error
    assert len(df.columns) == n_sens+2

    # drop timedelta column
    df = df.iloc[:, :-1]

    # drop outliers (all pressures greater than 115% of the median value and lower than 85 % of the median value)
    # Define the criteria for outliers
    lower_threshold = 0.85
    upper_threshold = 1.07

    # Iterate through each column to identify and filter out outliers
    for column in df.columns[1:]:
        median_value = df[column].median()
        lower_bound = lower_threshold * median_value
        upper_bound = upper_threshold * median_value
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # assign column names
    df.columns = columns

    # drop lines with missing data
    df = df.dropna().reset_index(drop=True)

    # remove outliers
    #df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)].reset_index(drop=True)

    # Calculate the time difference in milliseconds from the first row
    time_diff_ms = df['Time'] - df['Time'].iloc[0]

    # The maximum value pandas can handle (ns precision is default, ~year 2262)
    max_ms = pd.Timestamp.max.value // 10 ** 6  # Convert from ns to ms
    min_ms = pd.Timestamp.min.value // 10 ** 6

    series_ms = start_time_ms + time_diff_ms
    series_ms = np.clip(series_ms, min_ms, max_ms)

    # Remove or flag the out-of-range values
    valid = (series_ms >= min_ms) & (series_ms <= max_ms)
    df = df[valid]
    series_ms = series_ms[valid]

    if not valid.all():
        print(f"{(~valid).sum()} values are out of bounds for pandas datetime.")

    # Add this difference to the start time (in milliseconds) and convert back to datetime
    df['Time'] = pd.to_datetime(series_ms, unit='ms')

    # convert to UTC
    df['Time'] = (
        df['Time']
        .dt.tz_localize('UTC')  # Convert to UTC
    )

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
    start = merged_df.loc[0, 'Time']
    end = merged_df.loc[len(merged_df.index)-1, 'Time']
    merged_df = merged_df.sort_values(by="Time", ignore_index=True)
    merged_df = merged_df.loc[(merged_df.Time >= start) & (merged_df.Time <= end)]
    for df in merge_dfs_list[1:]:
        if len(df.index) > 0:
            start = df.loc[0, 'Time']
            end = df.loc[len(df.index)-1, 'Time']
            df = df.sort_values(by="Time", ignore_index=True)
            df = df.loc[(df.Time >= start) & (df.Time <= end)]
            merged_df = pd.merge_asof(merged_df, df.sort_values(by="Time", ignore_index=True), on='Time',
                                      tolerance=pd.Timedelta('1ms'), direction='nearest')

    # Set the index to 't_abs' to use time-based interpolation
    merged_df.set_index('Time', inplace=True)

    # Interpolate missing values using time-based interpolation
    merged_df = merged_df.interpolate(method='time')

    # localize index of df_sync to UTC
    merged_df.index = merged_df.index.tz_convert("UTC")

    return merged_df

def filter_data(df_sync):
    """
    applies savitzky golay filter to pressure data
    :param df_sync:
    :return:
    """
    # Build pattern and find matching columns
    patterns = [
        r'^pstat_rake_\d+$',
        r'^ptot_rake_\d+$',
        r'^static_K02_\d+$',
        r'^static_K03_\d+$',
        r'^static_K04_\d+$'
    ]
    combined_pattern = '|'.join(patterns)
    columns_to_filter = [col for col in df_sync.columns if re.match(combined_pattern, col)]

    # Set filter parameters
    window_length = 201
    polyorder = 2

    # Apply filter
    for col in columns_to_filter:
        df_sync[col] = savgol_filter(df_sync[col], window_length=window_length, polyorder=polyorder)

    return df_sync

def read_airfoil_geometry(filename, c, foil_source, eta_TE_flap, eta_LE_flap, flap_pivots, pickle_file=""):
    """
    --> searchs for pickle file in WD, if not found it creates a new pickle file
    --> generates pandas DataFrame with assignment of sensor unit + port to measuring point from Excel and renames
    --> adds 's' positions of the measuring points from Excel (line coordinate around the profile,
        starting at trailing edge)
    --> reads 'Kommentar' column of excel a nd drops sensors with status 'inop'
    --> calculates x and y position of static pressure points with airfoil coordinate file
    --> calculates x_n and y_n normal vector, tangential to airfoil surface of static pressure points
        with airfoil coordinate file

    :param filename:            file name of Excel eg. "Messpunkte Demonstrator.xlsx".
    :param c:                   airfoil chord length
    :param foil_source:         string, path of airfoil coordinate file
    :param eta_TE_flap:            flap deflection angle
    :param pickle_file:         path to pickle file with airfoil information
    :return df_airfoil:         DataFrame with info described above
    """



    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            df, foil, eta_LE_flap_read, eta_TE_flap_read = pickle.load(file)

    if not os.path.exists(pickle_file) or eta_LE_flap_read != eta_LE_flap or eta_TE_flap_read != eta_TE_flap:
        # initialize airfoilTools object
        foil = at.Airfoil(foil_source)

        if eta_TE_flap != 0.0:
            foil.flap(xFlap=flap_pivots[1, 0], yFlap=flap_pivots[1, 1], etaFlap=eta_TE_flap)
        if eta_LE_flap != 0.0:
            foil.LEflap(xFlap=flap_pivots[0, 0], yFlap=flap_pivots[0, 1], etaFlap=eta_LE_flap)

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
                pickle.dump([df, foil, eta_LE_flap, eta_TE_flap], file)

    return df, foil

def calc_ptot_pstat(df, sensor_defective_mask, prandtl_data, total_ref_pressure_method="trimmed median"):
    """
    calculates total reference pressure from wake rake data by using an asymmetric trimmed mean of pressure sensor values
    (cutoff of 5%, i.e. highest pressure and 50% lowest pressures)
    :param df:
    :param sensor_defective_mask:
    :param prandtl_data:                dict with "unit name static", "i_sens_static", "unit name total" and
                                        "i_sens_total".
                                        This specifies the sensor units and the index of the sensors of the Prandtl
                                        probe total
                                        pressure sensor and the static pressure sensor
    :param total_ref_pressure_method:   str, determines method of total reference pressure calculation. One of:
                                        "trimmed median", "trimmed average", "single
    :return: 
    """

    # calculate averaged mean
    cols = [f'ptot_rake_{i}' for i in range(1, 33) if i not in np.array(sensor_defective_mask)+1]

    colname_total = prandtl_data['unit name total'] + '_' + str(prandtl_data['i_sens_total'])
    ptot_single = df[colname_total] # use single reference sensor

    if total_ref_pressure_method == "trimmed median":
        df["ptot"] = df[cols].apply(
        lambda row: trimmed_median(row, lower_frac=0.5, upper_frac=0.05), axis=1
    )
    elif total_ref_pressure_method == "trimmed average":
        df["ptot"] = df[cols].apply(
        lambda row: asymmetric_trim_mean(row, lower_frac=0.5, upper_frac=0.05), axis=1
    )
    elif total_ref_pressure_method == "single":
        df["ptot"] = ptot_single

    # set static pressure
    colname_static = prandtl_data['unit name static'] + '_' + str(prandtl_data['i_sens_static'])
    df["pstat"] = df[colname_static]

    return df

def calc_airspeed_wind(df, l_ref):
    """
    --> calculates wind component in free stream direction

    :param df:          pandas DataFrame containing 'U_CAS' and 'U_GPS' column
    :return: df         pandas DataFrame with wind component column
    """

    ptot = df["ptot"]
    pstat = df["pstat"]

    # density of air according to International Standard Atmosphere (ISA)
    rho_ISA = 1.225
    R_s = 287.0500676
    # calculate derived variables (dynamic viscosity). Has to be calculated online if we chose to add a temp sensor
    # Formula from https://calculator.academy/viscosity-of-air-calculator/
    mu = (1.458E-6 * df["T_air"] ** (3 / 2)) / (df["T_air"] + 110.4)

    df['U_CAS'] = np.sqrt(2 * (ptot - pstat) / rho_ISA)

    # calculate air speeds
    rho = pstat / (R_s * df["T_air"])
    df['U_TAS'] = np.sqrt(np.abs(2 * (ptot - pstat) / rho))
    df['Re'] = df['U_TAS'] * l_ref * rho / mu

    # calculating wind component in free stream direction
    #df['wind_component'] = df['U_TAS'] - df['U_GPS']

    return df

def calc_cp(df, pressure_data_ident_strings):
    """
    calculates pressure coefficient for each static port on airfoil

    :param df:                          pandas DataFrame with synchronized and interpolated measurement data
    :param pressure_data_ident_strings: list of strings, which are contained in column names, which identify
                                        pressure sensor data
    :return: df                         pandas DataFrame with pressure coefficient in "static_K0X_Y" columns for
                                        every
                                        measuring point
    """

    ptot = df["ptot"]
    pstat = df["pstat"]

    # column names of all pressure sensor data
    pressure_cols = []
    for string in pressure_data_ident_strings:
        pressure_cols += [col for col in df.columns if string in col]

    # apply definition of c_p
    df[pressure_cols] = df[pressure_cols].apply(lambda p_col: (p_col - pstat)/(ptot - pstat))

    df.replace([np.inf, -np.inf], 0., inplace=True)

    return df

def calc_cl_cm_cdp(df, df_airfoil, flap_pivots=[], lambda_wall=0., sigma_wall=0., xi_wall=0.):
    """

    :param df:
    :param df_airfoil:
    :param flap_pivots:     position of flap hinge; TE: one point, if TE and LE: two points
    :param lambda_wall:
    :param sigma_wall:
    :param xi_wall:
    :return:
    """

    # calculate tap normal vector components on airfoil surface projected to aerodynamic coordinate system
    n_proj_z = np.dot(df_airfoil[['x_n', 'y_n']].to_numpy(), np.array([-np.sin(np.deg2rad(df['alpha'])),
                                             np.cos(np.deg2rad(df['alpha']))])).T
    n_proj_x = np.dot(df_airfoil[['x_n', 'y_n']].to_numpy(), np.array([np.cos(np.deg2rad(df['alpha'])),
                                             np.sin(np.deg2rad(df['alpha']))])).T

    # assign tap index to sensor unit and sensor port
    sens_ident_cols = ["static_K0{0:d}_{1:d}".format(df_airfoil.loc[i, "Sensor unit K"],
                                                     df_airfoil.loc[i, "Sensor port"]) for i in df_airfoil.index[1:-1]]
    # calculate virtual pressure coefficient
    df["static_virtualTE_top"] = df["static_virtualTE_bot"] = (df[sens_ident_cols[0]] + df[sens_ident_cols[-1]])/2
    # re-arrange columns
    cols = df.columns.to_list()
    cols = cols[:3*32] + cols[-2:] + cols[3*32:-2]
    df = df[cols].copy()
    sens_ident_cols = ["static_virtualTE_top"] + sens_ident_cols + ["static_virtualTE_bot"]


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

    return df, sens_ident_cols

def calc_cd(df, l_ref, lambda_wall, sigma_wall, xi_wall, sensor_defective_mask):
    """

    :param df:
    :return:
    """

    h_stat = 100
    h_tot = 93

    z_stat = np.linspace(-h_stat / 2, h_stat / 2, 5, endpoint=True)
    z_tot = np.linspace(-h_tot / 2, h_tot / 2, 32, endpoint=True)
    # delete defective (leaky) sensors
    z_tot = np.delete(z_tot, sensor_defective_mask)

    cp_stat_raw = df.filter(regex='^pstat_rake_').to_numpy()
    cp_stat_int = interpolate.interp1d(z_stat, cp_stat_raw, kind="linear", axis=1)

    cp_stat = cp_stat_int(z_tot)

    cp_tot = df.filter(regex='^ptot_rake_').to_numpy()
    # it is assumed, that 0th sensor is defective (omit that value)
    cp_tot = np.delete(cp_tot, sensor_defective_mask, axis=1)

    # Measurement of Proﬁle Drag by the Pitot-Traverse Method
    d_cd_jones = 2 * np.sqrt(np.abs((cp_tot - cp_stat))) * (1 - np.sqrt(np.abs(cp_tot)))

    # integrate integrand with simpson rule
    cd = integrate.trapezoid(d_cd_jones, z_tot) * 1 / (l_ref*1000)

    # apply wind tunnel wall corrections
    cd = cd * (1 - 2 * lambda_wall * (sigma_wall + xi_wall))

    df["cd"] = cd

    return df

def apply_calibration_offset(filename, df):

    with open(filename, "rb") as file:
        calibr_data = pickle.load(file)

    l_ref = calibr_data[6]
    T_air = calibr_data[5] + 273.15

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

    df["T_air"] = T_air

    return df, l_ref

def apply_calibration_20sec(df, T_air):
    """
    uses first 20 seconds to calculate pressure sensor calibration offsets
    :param df:      pandas Dataframe with pressure data
    :param T_air:   air temperature
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

    # Apply air temperature
    if not "T_air" in df.columns:
        df["T_air"] = T_air

    return df

def apply_manual_calibration(df, calibration_filename="manual_calibration_data.p", T_air=288.15):
    """
    uses first 20 seconds to calculate pressure sensor calibration offsets
    :param df:
    :return:
    """

    with open(calibration_filename, "rb") as file:
        offsets = pickle.load(file)

    # Apply the calibration to the entire DataFrame
    df.loc[:, offsets.index] = df.loc[:, offsets.index] - offsets

    # Apply air temperature
    if not "T_air" in df.columns:
        df["T_air"] = T_air

    return df

def apply_manual_calibration2(df, start_time, end_time, plot_speed=False, T_air=288.15):
    """
    uses time interval specified by start_time and end_time to calculate pressure sensor calibration offsets
    :param df:
    :param start_time:
    :param end_time:
    :return:
    """

    # identify all pressure sensor columns
    cols_calibrate = [col for col in df.columns if "static_K" in col or "rake" in col]

    # select data be
    df_calib_calc = df.loc[(df.index >= start_time) & (df.index <= end_time), cols_calibrate]

    calibration_offsets = df_calib_calc.mean().mean() - df_calib_calc.mean(axis=0)

    df[cols_calibrate] = df[cols_calibrate] + calibration_offsets

    if plot_speed:
        fig, ax = plt.subplots()
        ax.plot(df.index, df["U_GPS"])
        ax.axvspan(start_time, end_time, color="gray", alpha=0.5)

    # Apply air temperature
    if not "T_air" in df.columns:
        df["T_air"] = T_air

    return df

def calc_wall_correction_coefficients(filepath, l_ref):
    """
    calculate wall correction coefficients according to
    Abbott and van Doenhoff 1945: Theory of Wing Sections
    and
    Althaus 2003: Tunnel-Wall Corrections at the Laminar Wind Tunnel
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

def plot_time_series(df, df_segments, U_cutoff=10, plot_pstat=False, plot_drive=False, i_seg_plot=None):
    """

    :param df_sync:
    :return:
    """

    # plot U_CAS over time
    """fig, ax, = plt.subplots()
    ax.plot(df_cp["U_CAS"])
    ax.set_xlabel("$Time$")
    ax.set_ylabel("$U_{CAS} [m/s]$")
    ax.set_title("$U_{CAS}$ vs. Time")
    ax.xaxis.set_major_formatter(DateFormatter("%M:%S"))
    ax.grid()
    for index, row in df_segments.iterrows():
        if index == i_seg_plot:
            color = "green"
        else:
            color = 'lightgray'
        ax.axvspan(row['start'], row['end'], color=color, alpha=0.5)"""

    # plot alpha, cl, cm, cmr over time
    fig, host = plt.subplots()
    # Create twin axes on the right side of the host axis
    ax_alpha = host.twinx()
    ax_Re = host.twinx()
    ax_cd = host.twinx()
    if plot_pstat:
        ax_pstat = host.twinx()
    if plot_drive:
        ax_drive = host.twinx()
    # Offset the right twin axes so they don't overlap
    ax_alpha.spines['right'].set_position(('outward', 120))
    ax_Re.spines['right'].set_position(('outward', 60))
    ax_cd.spines['right'].set_position(('outward', 0))
    axpos = 120
    if plot_pstat:
        axpos += 60
        ax_pstat.spines['right'].set_position(('outward', axpos))
    if plot_drive:
        axpos +=60
        ax_drive.spines['right'].set_position(('outward', axpos))

    # Set plot lines
    ax_alpha.plot(df.loc[df["U_CAS"] > U_cutoff].index, df.loc[df["U_CAS"] > U_cutoff, "alpha"], "k-", label=r"$\alpha$", zorder=5)
    ax_Re.plot(df.loc[df["U_CAS"] > U_cutoff].index, df.loc[df["U_CAS"] > U_cutoff, "Re"], "y-", label=r"$Re$", zorder=4)
    host.plot(df.loc[df["U_CAS"] > U_cutoff].index, df.loc[df["U_CAS"] > U_cutoff, "cl"], label="$c_l$", zorder=3)
    ax_cd.plot(df.loc[df["U_CAS"] > U_cutoff].index,
               df.loc[df["U_CAS"] > U_cutoff, "cd"], color="red", label="$c_d$", zorder=1)

    if plot_pstat:
        ax_pstat.plot(df.index, df["p_stat"], color="green", label="$p_{stat}$")

    if plot_drive:
        ax_drive.plot(df.index, df["Rake Position"], color="purple")

    for index, row in df_segments.iterrows():
        if index == i_seg_plot:
            color = "green"
        else:
            color = 'lightgray'
        host.axvspan(row['start'], row['end'], color=color, alpha=0.5)
        host.annotate("$i_{seg}=" + str(index) + "$", xy=(row['start'], 0))

    # Formatting the x-axis to show minutes and seconds
    host.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    # Setting labels
    host.set_xlabel("$Time[hh:mm:ss, UTC]$")
    ax_alpha.set_ylabel(r"$\alpha~\mathrm{[^\circ]}$")
    host.set_ylabel("$c_l$")
    ax_Re.set_ylabel("$Re$")
    ax_cd.set_ylabel("$c_d$")
    ax_cd.set_ylim([0., 0.035])
    host.set_ylim([0, 2])
    if plot_pstat:
        ax_pstat.set_ylabel("$p_{stat}~\mathrm{[Pa]}$")
    # Enabling grid on host
    host.grid()
    # Adding legends from all axes
    lines, labels = [], []
    axes = [host, ax_alpha, ax_Re, ax_cd]
    if plot_pstat:
        axes.append(ax_pstat)
    if plot_drive:
        axes.append(ax_drive)
    for ax in axes:
        line, label = ax.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)
    host.tick_params(axis='x', labelrotation=80)
    fig.legend(lines, labels, loc='upper right')
    plt.show(block=False)
    plt.pause(0.01)

    # plot path of car
    fig5, ax3 = plt.subplots()
    transformer = Transformer.from_crs("wgs84", "EPSG:3035")
    x, y = transformer.transform(df["Longitude"], df["Latitude"])
    ax3.plot(x, y, "k-")
    for index, row in df_segments.iterrows():
        x, y = transformer.transform(df.loc[((df.index >= row['start']) & (df.index <= row['end'])), "Longitude"],
                 df.loc[((df.index >= row['start']) & (df.index <= row['end'])), "Latitude"])
        ax3.plot(x, y)
        t_center = row['start'] + (row['end'] - row['start'])/2
        i_center = abs(df.index - t_center).argmin()
        x, y = transformer.transform(df.iloc[i_center].loc["Longitude"],
                                                         df.iloc[i_center].loc["Latitude"])
        ax3.annotate("$i_{seg}=" + str(index) + "$", xy=(x, y))

    return

def plot_cp_x_and_wake(df, df_airfoil, at_airfoil, sens_ident_cols, df_segments, df_polar, sensor_defective_mask):
    """
    plots cp(x) and wake depression (x) at certain operating points (alpha, Re and beta)
    :param df:      pandas dataframe with index time and data to be plotted
    :param t:       index number of operating point (=time)
    :return:
    """
    h_stat = 100
    h_tot = 93

    # positions of total pressure sensors of wake rake
    z_tot = np.linspace(-h_tot / 2, h_tot / 2, 32, endpoint=True);
    # it is assumed, that 0th sensor is defective (omit that value)
    z_tot = np.delete(z_tot, sensor_defective_mask)

    # positions of static pressure sensors of wake rake
    z_stat = np.linspace(-h_stat / 2, h_stat / 2, 5, endpoint=True);

    for i_seg in df_polar.index:

        Re = df_polar.loc[i_seg, "Re"]
        alpha = df_polar.loc[i_seg, "alpha"]
        cl = df_polar.loc[i_seg, "cl"]
        cd = df_polar.loc[i_seg, "cd"]


        t_start = np.abs(df_sync.index-df_segments.loc[i_seg, "start"]).argmin()
        t_end = np.abs(df_sync.index-df_segments.loc[i_seg, "end"]).argmin()

        # plot cp(x)
        fig, axes = plt.subplots(2, figsize=(6,12))
        ax = axes[0]
        ax_cp = ax.twinx()
        ax.plot(at_airfoil.coords[:, 0], at_airfoil.coords[:, 1], "k-")
        ax.plot(df_airfoil["x"], df_airfoil["y"], "k.")
        ax_cp.plot(df_airfoil["x"], df[sens_ident_cols].iloc[t_start:t_end].mean(), "r.-")
        # Calculate mean values and standard deviations over the specified time interval
        mean_cp_values = df[sens_ident_cols].iloc[t_start:t_end].mean()
        std_cp_values = df[sens_ident_cols].iloc[t_start:t_end].std()
        # Plot the mean cp values with error bars
        ax_cp.errorbar(df_airfoil["x"], mean_cp_values, yerr=std_cp_values, fmt='r.-', ecolor='gray', elinewidth=1,capsize=2)
        ylim_u, ylim_l = ax_cp.get_ylim()
        ax_cp.set_ylim([ylim_l, ylim_u])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax_cp.set_ylabel("$c_p$")
        fig.suptitle(r"$i_\mathrm{{seg}}={}:~Re={:.2E}~cl={:.2f}~cd={:.4f}~\alpha={:.2f}$".format(i_seg, Re, cl, cd, alpha))
        ax_cp.grid()
        ax.axis("equal")

        # plot wake depression(x)
        # extract total pressure of wake rake from dataframe
        cols = df.filter(regex='^ptot_rake')

        # drop erroneous pressure sensors
        cols_tot = cols.drop(cols.columns[sensor_defective_mask], axis=1)
        # data of static pressure sensor
        cols_stat = df.filter(regex='^pstat_rake')
        ax = axes[1]
        ax_cp = ax.twiny()
        # plot airfoil for visualization
        ax.plot(at_airfoil.coords[:, 0]*100, at_airfoil.coords[:, 1]*100, "k-")
        # Calculate mean and standard deviation over the specified time interval
        mean_ptot_values = cols_tot.iloc[t_start:t_end].mean()
        std_ptot_values = cols_tot.iloc[t_start:t_end].std()
        mean_pstat_values = cols_stat.iloc[t_start:t_end].mean()
        std_pstat_values = cols_stat.iloc[t_start:t_end].std()
        # Plot the mean ptot values with error bars
        ax_cp.plot(mean_ptot_values, z_tot, "r.-")
        ax_cp.errorbar(mean_ptot_values, z_tot, xerr=std_ptot_values, fmt='r.-', ecolor='gray', elinewidth=1, capsize=2)
        # Plot the mean pstat values with error bars
        ax_cp.plot(mean_pstat_values, z_stat, "b.-")
        ax_cp.errorbar(mean_pstat_values, z_stat, xerr=std_pstat_values, fmt='b.-', ecolor='gray', elinewidth=1, capsize=2)
        ylim_l, ylim_u = ax_cp.get_ylim()
        ax_cp.set_ylim([ylim_l, ylim_u])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$z$")
        ax_cp.set_xlabel("$c_p$")
        ax_cp.grid()
        ax.axis("equal")
        fig.tight_layout()
        plt.pause(0.01)

        # write cp to file
        filename = "C:/XFOIL6.99/{0}_Re{1:.2E}_cl{2:.2f}_alpha{3:.2f}.cp".format(at_airfoil.filename.split(".dat")[0], Re, cl, alpha)
        cp_offset = 0.15
        x_vals = df_airfoil["x"].to_numpy()
        cp_vals = df[sens_ident_cols].iloc[t_start:t_end].mean()
        x_cp = np.vstack((x_vals, cp_vals + cp_offset)).T

        np.savetxt(filename, x_cp, header="     x          Cp  ", fmt="%.5f")

    return

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


    return

def calc_mean(df, alpha, Re):

    """calculates mean values of AOA, lift-, drag- and moment coefficients for a given alpha (automativally given from
    calc_means when called) and an entered Reynoldsnumber
    :param df:          pandas dataframe with column names: alpha, cl, cd, cm (given from plot polars function)
    :param alpha:       automativally given from calc_means when called
    :param Re:          desired Reynoldsnumber for the test (needs to be typed in function call)
    :return:"""
    
    # define Intervalls (might be adapted)
    delta_alpha = 0.09
    min_alpha = alpha - delta_alpha
    max_alpha = alpha + delta_alpha
    delta_Re = 0.2e6
    min_Re = Re - delta_Re
    max_Re = Re + delta_Re

    # conditions for representative values
    condition = ((df["alpha"] > min_alpha) &
                 (df["alpha"] < max_alpha) &
                 (df["Re"] > min_Re) &
                 (df["Re"] < max_Re))# &
                 #(df["Rake Speed"] != 0))



    # pick values which fulfill the condition
    col_alpha = df.loc[condition, "alpha"]
    col_cl = df.loc[condition, "cl"]
    col_cd = df.loc[condition, "cd"]
    col_cm = df.loc[condition, "cm"]

    # calculate mean values
    mean_alpha = col_alpha.mean()
    mean_cl = col_cl.mean()
    mean_cd = col_cd.mean()
    mean_cm = col_cm.mean()

    return mean_alpha, mean_cl, mean_cd, mean_cm

def calculate_polar(df_raw, df_segments, prandtl_data, df_airfoil, l_ref, flap_pivots, lambda_wall, sigma_wall,
                    xi_wall, sensor_defective_mask=(), total_ref_pressure_method="trimmed average"):
    """
    calculates the polars
    :param df_raw:      synchronized raw data
    :param df_segments: pandas DataFrame with measurement segments start and end time
    :param prandtl_data:                dict with "unit name static", "i_sens_static", "unit name total" and
                                        "i_sens_total".
                                        This specifies the sensor units and the index of the sensors of the Prandtl
                                        probe total
                                        pressure sensor and the static pressure sensor
    :param df_airfoil:                  DataFrame with airfoil information: location and normal vectors of pressure tabs
    :param l_ref:                       float, reference length (i.e. chord length of the airfoil) in meters
    :param flap_pivots:                 2x2 numpy.ndarray with positions of flap hinges: First line: leading edge flap,
                                        second line: trailing edge flap
    :param lambda_wall:                 wall correction coefficient lambda (see Döller 2016)
    :param sigma_wall:                  wall correction coefficient sigma (see Döller 2016)
    :param xi_wall:                     wall correction coefficient xi (see Döller 2016)
    :param sensor_defective_mask:       list of ints with indices of defective sensors. Will be omitted from drag calc
    :param total_ref_pressure_method:   str, determines method of total reference pressure calculation. One of:
                                        "trimmed median", "trimmed average", "single
    :return: polar
    """

    #Average data over segments first
    data = []
    for i in range(len(df_segments.index)):
        start_time = df_segments.loc[i, "start"]
        end_time = df_segments.loc[i, "end"]
        df_seg = df_raw.loc[(df_raw.index >= start_time) & (df_raw.index <= end_time), :]
        s_seg = df_seg.mean()
        data.append(s_seg)
    df_polar = pd.DataFrame(data, columns=df_raw.columns)

    # calculate total and static pressures
    df_polar = calc_ptot_pstat(df_polar, sensor_defective_mask, prandtl_data, total_ref_pressure_method)

    # calculate airspeed and wind component
    df_polar = calc_airspeed_wind(df_polar, l_ref)

    # calculate pressure coefficients from absolute pressures
    df_polar = calc_cp(df_polar, pressure_data_ident_strings=['stat', 'ptot'])

    # calculate lift coefficients
    df_polar, _ = calc_cl_cm_cdp(df_polar, df_airfoil, flap_pivots, lambda_wall, sigma_wall, xi_wall)

    # calculate drag coefficients
    df_polar = calc_cd(df_polar, l_ref, lambda_wall, sigma_wall, xi_wall, sensor_defective_mask)

    return df_polar

def plot_polars(df):
    """

    :param df_polars:
    :return:
    """
    # plot cl(alpha)
    fig, ax = plt.subplots()
    ax.plot(df["alpha"], df["cl"], "k.", linestyle='-')
    ax.set_xlabel(r"$\alpha$")
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
    ax4.set_xlabel(r"$\alpha$")
    ax4.set_ylabel("$c_m$")
    ax4.set_title("$c_m$ vs. alpha")
    ax4.grid()



    return


if __name__ == '__main__':

    plot = True

    T_air = 288
    # Lower cutoff speed for plots
    U_cutoff = 10
    # specify test segment, which should be plotted
    i_seg_plot = 0

    # lower and upper cutoff fraction for asymmetric trimmed mean calculation for reference total pressure
    lower_frac_ptot = 0.05
    upper_frac_ptot = 0.5

    PPAX = dict()
    PPAX['CLmin'] = 0.0
    PPAX['CLmax'] = 2.000
    PPAX['CLdel'] = 0.5000
    PPAX['CDmin'] = 0.0000
    PPAX['CDmax'] = 0.0250
    PPAX['CDdel'] = 0.0050
    PPAX['ALmin'] = 0.0000
    PPAX['ALmax'] = 15.0000
    PPAX['ALdel'] = 2.0000
    PPAX['CMmin'] = -0.2500
    PPAX['CMmax'] = 0.000
    PPAX['CMdel'] = 0.0500


    airfoil = "Mü13-33"
    #airfoil = "B200"
    #airfoil = "B200_topseal"
    # constants and input data
    if airfoil == "Mü13-33":
        #run = "T007"
        run = "T008"

        if run == "T007":
            calibration_start_time =  "2024-06-13 22:01:26"
            calibration_end_time =  "2024-06-13 22:01:47"
            # Parse into datetime objects
            calibration_start_time = datetime.strptime(calibration_start_time, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc)
            calibration_end_time = datetime.strptime(calibration_end_time, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc)
            seg_def_files = ["T007.xlsx"]
            digitized_LWK_polar_files_clcd = ["Re1e6_cl-cd.txt"]
            digitized_LWK_polar_files_clalpha = ["Re1e6_cl-alpha.txt"]
            if os.getlogin() == 'joeac':
                WDIR = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/Mü13-33/2024-06-13/T002_T009"

        if run == "T008":
            calibration_start_time = "2024-06-13 22:37:30"
            calibration_end_time = "2024-06-13 22:38:00"
            # Parse into datetime objects
            calibration_start_time = datetime.strptime(calibration_start_time, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc)
            calibration_end_time = datetime.strptime(calibration_end_time, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc)
            seg_def_files = ["T008_09.xlsx"]
            digitized_LWK_polar_files_clcd = ["Re1e6_cl-cd.txt"]
            digitized_LWK_polar_files_clalpha = ["Re1e6_cl-alpha.txt"]
            if os.getlogin() == 'joeac':
                WDIR = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/Mü13-33/2024-06-13/T002_T009"

        # set calibration type in seg_def Excel file ("20sec", "manual", "file")
        # set flap deflection in seg_def Excel file
        if os.getlogin() == 'joeac':
            segments_def_dir = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/Mü13-33/testsegments_specification"
            digitized_LWK_polar_dir = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/Mü13-33/Digitized data Döller LWK/"
            ref_dat_path = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/Mü13-33/01_Reference Data/"
        else:
            WDIR = "D:/Python_Codes/Workingdirectory_Auswertung"
            segments_def_dir = "D:/Python_Codes/Rohdateien/Zeitabschnitte_Polaren"
            digitized_LWK_polar_dir = "D:/Python_Codes/Rohdateien/digitized_polars_doeller"
            ref_dat_path = "D:/Python_Codes/Workingdirectory_Auswertung/"

        prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                        #"unit name total": "ptot_rake", "i_sens_total": 3} # This is the third sensor in the wake rake
                        "unit name total": "static_K04", "i_sens_total": 32} # This is the original total pressure sensor

        XFOIL_polar_files = []
        foil_coord_path = os.path.join(ref_dat_path, "mue13-33-le15.dat")
        file_path_msr_pts = os.path.join(ref_dat_path, 'Messpunkte Demonstrator_Mue13-33.xlsx')
        pickle_path_msr_pts = os.path.join(ref_dat_path, 'Messpunkte Demonstrator.p')
        cp_path_wall_correction = os.path.join(ref_dat_path, 'mue13-33-le15-tgap0_14.cp')
        # alpha sensor offset
        alpha_sens_offset = 214.73876953125

        sensor_defective_mask = [0, ]
        l_ref = 0.7
        flap_pivots = np.array([[0.2, 0.0], [0.8, 0.0]])
        eta_LE_flap = 0.0
        # specifiy, if drive data should be synchronized
        sync_drive = False
        # Raw data file prefix

    elif airfoil == "B200":

        run = "T006"
        #run = "T010"
        #run = "T012"

        l_ref = 0.5
        flap_pivots = np.array([[0.325, 0.0], [0.87, -0.004]])
        # specifiy, if drive data should be synchronized
        sync_drive = True

        if run == "T006":
            seg_def_files = ["T006.xlsx"]
            XFOIL_polar_files = ["B200-0_reinit_Re11e5_XFOILSUC.pol"]
            WDIR = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/2023_09_26/T6"
            sensor_defective_mask = [0, 24]
            prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                            "unit name total": "ptot_rake", "i_sens_total": 3}
            # "unit name total": "static_K04", "i_sens_total": 32}

        if run == "T010":
            seg_def_files = ["T010.xlsx"]
            XFOIL_polar_files = ["B200_5deg_reinitialized_Re1_08.pol"]
            WDIR = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/2023_09_26/T10_R012"
            sensor_defective_mask = [0, 24]
            prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                            "unit name total": "ptot_rake", "i_sens_total": 30}
            # "unit name total": "static_K04", "i_sens_total": 32}

        if run == "T012":
            seg_def_files = ["T012.xlsx"]
            XFOIL_polar_files = ["B200-1_reinit_Re1e6_XFOIL_HLIDP.pol", "B200-1_reinit_Re1e6_XFOIL_HLIDP_xtr0_325.pol"]
            WDIR = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/2023_08_03/R003_20deg_clmax"
            sensor_defective_mask = [0, 24]
            prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                            "unit name total": "ptot_rake", "i_sens_total": 3}
            # "unit name total": "static_K04", "i_sens_total": 32}

        digitized_LWK_polar_files_clcd = []
        digitized_LWK_polar_files_clalpha = []
        segments_def_dir = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/Testsegments_specification"
        digitized_LWK_polar_dir = ""
        ref_dat_path = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/01_Reference Data/"
        prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                        "unit name total": "ptot_rake", "i_sens_total": 3}
                        #"unit name total": "static_K04", "i_sens_total": 32}

        foil_coord_path = os.path.join(ref_dat_path, "B200-0_reinitialized.dat")
        file_path_msr_pts = os.path.join('C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/03_Static pressure measurement system/Messpunkte Demonstrator/Messpunkte Demonstrator.xlsx')
        pickle_path_msr_pts = os.path.join(ref_dat_path, 'Messpunkte Demonstrator.p')
        cp_path_wall_correction = os.path.join(ref_dat_path, 'B200-0_reinitialized.cp')
        # alpha sensor offset
        alpha_sens_offset = 162.88330078125 # Campaign 1

    elif airfoil == "B200_topseal":
        #run = "T010_R23"
        #run = "T006_R24"
        #run = "T006_R26"
        run = "T012_R27"

        l_ref = 0.5
        flap_pivots = np.array([[0.325, 0.09], [0.87, -0.004]])
        # specifiy, if drive data should be synchronized
        sync_drive = True

        if run == "T010_R23":
            sensor_defective_mask = [0, 24, 30]
            seg_def_files = ["T010_R023.xlsx"]
            XFOIL_polar_files = ["B200-LE2deg_reinitialized_TE0_from1_Re1e6_XFOILSUC.pol"]
            WDIR = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/2025_05_20/R023/"
            prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                            "unit name total": "ptot_rake", "i_sens_total": 2}
                            # "unit name total": "static_K04", "i_sens_total": 32}

        if run == "T006_R24":
            sensor_defective_mask = [0, 8, 24, 30]
            seg_def_files = ["T006_R024.xlsx"]
            XFOIL_polar_files = ["B200-0_Re1e6_XFOILSUC.pol", "B200-0_reinitialized_from1_Re1e6_XFOILSUC.pol"]
            WDIR = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/2025_05_20/R024/"
            prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                            "unit name total": "ptot_rake", "i_sens_total": 2}
                            # "unit name total": "static_K04", "i_sens_total": 32}

        if run == "T006_R26":
            sensor_defective_mask = [0, 24, 30]
            seg_def_files = ["T006_R026.xlsx"]
            XFOIL_polar_files = ["B200-0_reinitialized_from1_Re75e4_XFOILSUC.pol", "B200-0_reinitialized_from1_Re75e4_XFOIL_mod.pol"]
            WDIR = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/2025_05_20/R026/"
            prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                            "unit name total": "ptot_rake", "i_sens_total": 2}
                            # "unit name total": "static_K04", "i_sens_total": 32}

            PPAX['ALmin'] = -2.0000
            PPAX['ALmax'] = 18.0000
            PPAX['ALdel'] = 5.0000

        if run == "T012_R27":
            sensor_defective_mask = [0, 1, 24, 25, 30]
            seg_def_files = ["T012_R027.xlsx"]
            XFOIL_polar_files = ["B200-1_xtr0_325_Re95e4_XFOIL_HLIDP.pol", "B200-1_xtrb0_325_Re95e4_XFOIL_HLIDP.pol", "B200-1_xtrt0_325_Re95e4_XFOIL_HLIDP.pol", "B200-1_Re95e4_XFOIL_HLIDP.pol", "B200-1_Re95e4_XFOILSUC_mod.pol"]
            WDIR = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/2025_05_20/R027/"
            prandtl_data = {"unit name static": "static_K04", "i_sens_static": 31,
                            "unit name total": "ptot_rake", "i_sens_total": 2}
                            # "unit name total": "static_K04", "i_sens_total": 32}

            PPAX['ALmin'] = -2.0000
            PPAX['ALmax'] = 18.0000
            PPAX['ALdel'] = 5.0000

        digitized_LWK_polar_files_clcd = []
        digitized_LWK_polar_files_clalpha = []
        segments_def_dir = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/Testsegments_specification"
        digitized_LWK_polar_dir = ""
        ref_dat_path = "C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/07_Results/B200/01_Reference Data/"


        foil_coord_path = os.path.join(ref_dat_path, "B200-0_reinitialized.dat")
        file_path_msr_pts = os.path.join('C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/ALF - General/Auto-Windkanal/03_Static pressure measurement system/Messpunkte Demonstrator/Messpunkte Demonstrator-17.05.2025.xlsx')
        pickle_path_msr_pts = os.path.join(ref_dat_path, 'Messpunkte Demonstrator.p')
        cp_path_wall_correction = os.path.join(ref_dat_path, 'B200-0_reinitialized.cp')

        alpha_sens_offset = 272.96630859375  # Campaign 2

    #******************************************************************************************************************
    #******************************************************************************************************************
    #******************************************************************************************************************
    #******************************************************************************************************************

    os.chdir(WDIR)

    list_of_df_polars = ([])
    list_of_polars = []
    list_of_eta_flaps = []

    for i_file, seg_def_file in enumerate(seg_def_files):

        digitized_LWK_polar_paths = []
        for i in range(len(digitized_LWK_polar_files_clcd)):
            digitized_LWK_polar_paths.append([os.path.join(digitized_LWK_polar_dir, digitized_LWK_polar_files_clcd[i]),
                                              os.path.join(digitized_LWK_polar_dir, digitized_LWK_polar_files_clalpha[i])])

        # get segments filenames
        segments_def_path = os.path.join(segments_def_dir, seg_def_file)


        # read raw data filenames
        raw_data_filenames = pd.read_excel(segments_def_path, skiprows=0, usecols="J").dropna().values.astype(
            "str").flatten()
        calibration_types = pd.read_excel(segments_def_path, skiprows=0, usecols="K").dropna().values.astype(
            "str").flatten()
        etas_TE_flap = pd.read_excel(segments_def_path, skiprows=0, usecols="L").dropna().values.astype(
            "float").flatten()
        etas_LE_flap = pd.read_excel(segments_def_path, skiprows=0, usecols="M").dropna().values.astype(
            "float").flatten()

        # read segment times
        df_segments = pd.read_excel(segments_def_path, skiprows=1, usecols="A:H").ffill(axis=0)
        df_segments[["hh", "mm", "ss", "hh.1", "mm.1", "ss.1"]] = df_segments[["hh", "mm", "ss", "hh.1", "mm.1", "ss.1"]].astype(int)

        df_segments['start'] = pd.to_datetime(df_segments['dd'].astype(str) + ' ' +
                                              df_segments['hh'].astype(str) + ':' +
                                              df_segments['mm'].astype(str) + ':' +
                                              df_segments['ss'].astype(str),
                                              errors='coerce', utc=True)
        df_segments['end'] = pd.to_datetime(df_segments['dd.1'].astype(str) + ' ' +
                                            df_segments['hh.1'].astype(str) + ':' +
                                            df_segments['mm.1'].astype(str) + ':' +
                                            df_segments['ss.1'].astype(str),
                                            errors='coerce', utc=True)

        df_segments = df_segments[['start', 'end']]

        df_sync = pd.DataFrame()
        list_of_dfs = []

        for i, filename in enumerate(raw_data_filenames):
            eta_TE_flap = etas_TE_flap[i]
            list_of_eta_flaps.append(eta_TE_flap)
            if len(etas_LE_flap) == len(raw_data_filenames):
                eta_LE_flap = etas_LE_flap[i]
            elif etas_LE_flap.size > 0:
                eta_LE_flap = etas_LE_flap[0]
            else:
                eta_LE_flap = 0

            # read airfoil data (only in case of first file or if flap setting changes)
            if i == 0 or not (etas_TE_flap == etas_TE_flap[0]).all() or len(np.unique(etas_LE_flap, return_counts=True)[0]) <= 1:
                df_airfoil, at_airfoil = read_airfoil_geometry(file_path_msr_pts, c=l_ref, foil_source=foil_coord_path,
                                                            eta_LE_flap=eta_LE_flap, eta_TE_flap=eta_TE_flap,
                                                            flap_pivots=flap_pivots, pickle_file=pickle_path_msr_pts)

            # calculate wall correction coefficients
            lambda_wall, sigma_wall, xi_wall = calc_wall_correction_coefficients(cp_path_wall_correction, l_ref)

            calibration_type = calibration_types[i]
            if calibration_type not in ["file", "20sec", "manual2"]:
                calibration_filename = calibration_type
                calibration_type = "manual"

            file_path_drive = os.path.join(WDIR, f"{filename}_drive.dat")
            file_path_AOA = os.path.join(WDIR, f"{filename}_AOA.dat")
            file_path_pstat_K02 = os.path.join(WDIR, f"{filename}_static_K02.dat")
            file_path_pstat_K03 = os.path.join(WDIR, f"{filename}_static_K03.dat")
            file_path_pstat_K04 = os.path.join(WDIR, f"{filename}_static_K04.dat")
            file_path_ptot_rake = os.path.join(WDIR, f"{filename}_ptot_rake.dat")
            file_path_pstat_rake = os.path.join(WDIR, f"{filename}_pstat_rake.dat")
            file_path_GPS = os.path.join(WDIR, f"{filename}_GPS.dat")
            pickle_path_calibration = os.path.join(WDIR, f"{filename}_sensor_calibration_data.p")

            # read sensor data
            GPS = read_GPS(file_path_GPS)

            alphas, delta_t_GPS_PC = read_AOA_file(file_path_AOA, sigma_wall, t0=GPS["Time"].iloc[0], alpha_sens_offset=alpha_sens_offset)
            pstat_K02 = read_DLR_pressure_scanner_file(file_path_pstat_K02, n_sens=32, t0=GPS["Time"].iloc[0])
            pstat_K03 = read_DLR_pressure_scanner_file(file_path_pstat_K03, n_sens=32, t0=GPS["Time"].iloc[0])
            pstat_K04 = read_DLR_pressure_scanner_file(file_path_pstat_K04, n_sens=32, t0=GPS["Time"].iloc[0])
            ptot_rake = read_DLR_pressure_scanner_file(file_path_ptot_rake, n_sens=32, t0=GPS["Time"].iloc[0])
            pstat_rake = read_DLR_pressure_scanner_file(file_path_pstat_rake, n_sens=5, t0=GPS["Time"].iloc[0])

            if sync_drive:
                drive = read_drive(file_path_drive, t0=GPS["Time"].iloc[0], delta_t=delta_t_GPS_PC)

            # synchronize sensor data
            if sync_drive:
                sync_data = [pstat_K02, pstat_K03, pstat_K04, ptot_rake, pstat_rake, alphas, drive, GPS]
            else:
                sync_data = [pstat_K02, pstat_K03, pstat_K04, ptot_rake, pstat_rake, alphas, GPS]
            df_sync = synchronize_data(sync_data)

            if calibration_type == "file":
                # apply calibration offset from calibration file
                df_sync, l_ref = apply_calibration_offset(pickle_path_calibration, df_sync)
            elif calibration_type == "20sec":
                # apply calibration offset from first 20 seconds
                df_sync = apply_calibration_20sec(df_sync, T_air)
            elif calibration_type == "manual":
                df_sync = apply_manual_calibration(df_sync, calibration_filename="manual_calibration_data.p")
            elif calibration_type == "manual2":
                # manual2 is performed after merging the dataframes
                pass
            else:
                raise ValueError("wrong parameter 'calibration_type' passed. Either 'file', '20sec', 'manual' or 'manual2'")

            # append the processed data to the all_data DataFrame
            list_of_dfs.append(df_sync)
        if len(raw_data_filenames) > 1:
            df_sync = pd.concat(list_of_dfs)

        # perform calibration with
        if calibration_type == "manual2":
            df_sync = apply_manual_calibration2(df_sync, calibration_start_time, calibration_end_time)

        df_raw = df_sync.copy()

        # filter data
        df_filt = filter_data(df_raw.copy())

        # calculate total reference pressure
        df_raw = calc_ptot_pstat(df_raw, sensor_defective_mask, prandtl_data, total_ref_pressure_method="trimmed median")
        df_filt = calc_ptot_pstat(df_filt, sensor_defective_mask, prandtl_data, total_ref_pressure_method="trimmed median")

        # calculate wind component
        df_raw = calc_airspeed_wind(df_raw, l_ref)
        df_filt = calc_airspeed_wind(df_filt, l_ref)

        # calculate pressure coefficients
        df_raw = calc_cp(df_raw, pressure_data_ident_strings=['stat', 'ptot'])
        df_filt = calc_cp(df_filt, pressure_data_ident_strings=['stat', 'ptot'])

        # calculate lift coefficients
        df_raw, _ = calc_cl_cm_cdp(df_raw, df_airfoil, flap_pivots, lambda_wall, sigma_wall, xi_wall)
        df_filt, sens_ident_cols = calc_cl_cm_cdp(df_filt, df_airfoil, flap_pivots, lambda_wall, sigma_wall,
                                                      xi_wall)

        # calculate drag coefficients
        df_filt = calc_cd(df_filt, l_ref, lambda_wall, sigma_wall, xi_wall, sensor_defective_mask)

        # visualisation of time series
        if plot:
            plot_time_series(df_filt, df_segments, U_cutoff, plot_drive=sync_drive, i_seg_plot=i_seg_plot)

        # generate the polar
        ptot_method = "trimmed average"
        df_polar = calculate_polar(df_sync, df_segments, prandtl_data, df_airfoil, l_ref, flap_pivots, lambda_wall,
                                       sigma_wall, xi_wall, sensor_defective_mask,
                                       total_ref_pressure_method=ptot_method)
        df_polar_result_only = df_polar.loc[:, ['alpha', 'Re', 'cl', 'cd', 'cdp', 'U_CAS', 'U_TAS', 'cm',
       'cmr_LE', 'cmr_TE',]]
        list_of_df_polars.append(df_polar)

        # plot cp(x) and cp wake
        if plot:
            plot_cp_x_and_wake(df_raw, df_airfoil, at_airfoil, sens_ident_cols, df_segments, df_polar, sensor_defective_mask)

        # Generate PolarTool polar
        Re_mean = np.around(df_polar.loc[:, "Re"].mean() / 5e4)*5e4
        polar = at.PolarTool(name="Car-mounted aerodynamic measurement platform", Re=Re_mean, flapangle=eta_TE_flap,
                             WindtunnelName="MoProMa-Car")
        polar.parseMoProMa_Polar(df_polar)
        list_of_polars.append(polar)

        polar.writeXFoilPol("C:/XFOIL6.99", "MoProMa_Polar.pol")

    # read measured polar from LWK Stuttgart, digitized with getData graph digitizer
    polarsStu = list()
    for (path_clcd, path_clalpha), eta_TE_flap in zip(digitized_LWK_polar_paths, list_of_eta_flaps):
        polarsStu.append(at.PolarTool(name="LWK Stuttgart", Re=Re_mean, flapangle=eta_TE_flap))
        polarsStu[-1].read_getDataGraphDigitizerPolar(path_clcd, path_clalpha)

    # read XFOIL polars
    XFOIL_polars = list()
    for XFOIL_polar_file in XFOIL_polar_files:
        XFOIL_polars.append(at.PolarTool(name="XFOILSUC-mod", flapangle=eta_TE_flap))
        XFOIL_polars[-1].ImportXFoilPolar(os.path.join(ref_dat_path, XFOIL_polar_file), drag_correction_factor=1.12)
        if "XFOIL_HLIDP" in XFOIL_polar_file:
            XFOIL_polars[-1].name = "XFOIL-mod"
        elif "XFOILSUC" in XFOIL_polar_file:
            XFOIL_polars[-1].name = "XFOILSUC-mod"

    LineAppearance = dict()

    LineAppearance['color'] = []
    LineAppearance['linestyle'] = []
    LineAppearance['marker'] = []
    # R G B
    LineAppearance['color'].append((255. / 255., 68. / 255., 68. / 255.))  # red
    #LineAppearance['color'].append("k")
    LineAppearance['color'].append((255. / 255., 165. / 255., 0. / 255.))  # orange
    #LineAppearance['color'].append("k")
    LineAppearance['color'].append((255. / 255., 255. / 255., 68. / 255.))  # yellow
    LineAppearance['color'].append("k")
    LineAppearance['color'].append((68. / 255., 255. / 255., 68. / 255.))  # green
    LineAppearance['color'].append((68. / 255., 255. / 255., 255. / 255.))  # turquoise
    LineAppearance['color'].append((60. / 255., 155. / 255., 255. / 255.))  # light blue
    LineAppearance['color'].append((205. / 255., 55. / 255., 255. / 255.))  # purple
    LineAppearance['color'].append((255. / 255., 0. / 255., 255. / 255.))  # rose/purple

    LineAppearance['linestyle'].append("None")
    LineAppearance['linestyle'].append("-")
    LineAppearance['linestyle'].append("None")
    LineAppearance['linestyle'].append("-")
    LineAppearance['linestyle'].append("None")
    LineAppearance['linestyle'].append("-")
    LineAppearance['linestyle'].append("None")
    LineAppearance['linestyle'].append("-")

    LineAppearance['marker'].append("+")
    LineAppearance['marker'].append("^")
    LineAppearance['marker'].append("s")
    LineAppearance['marker'].append("s")
    LineAppearance['marker'].append('o')
    LineAppearance['marker'].append('o')
    LineAppearance['marker'].append('o')
    LineAppearance['marker'].append('x')
    LineAppearance['marker'].append('x')

    altsort_polars = []
    i_line = 0
    for a, b, c in itertools.zip_longest(list_of_polars, polarsStu, XFOIL_polars):
        if a:
            altsort_polars.append(a)
            i_line += 1
        if b:
            altsort_polars.append(b)
            i_line += 1
        if c:
            altsort_polars.append(c)
            # do not show marker for XFOIL polars (makes plot unreadable due to tight operation point spacing)
            LineAppearance['marker'][i_line] = "None"
            LineAppearance['linestyle'][i_line] = "-"
            i_line += 1

    if plot:
        altsort_polars[0].plotPolar(additionalPolars=altsort_polars[1:], PPAX=PPAX, Colorplot=True,
                                    LineAppearance=LineAppearance, highlight=i_seg_plot)
    altsort_polars[0].plotPolar(additionalPolars=altsort_polars[1:], PPAX=PPAX, Colorplot=True,
                                LineAppearance=LineAppearance, saveFlag=True, format="pdf",
                                saveFileName=seg_def_files[0].rstrip(".xlsx"))

    # export polar
    if not "run" in locals():
        run = ""
    altsort_polars[0].writeXFoilPol("C:/XFOIL6.99", airfoil+".pol")

    print("done")



