from airfoilwinggeometry.AirfoilPackage import AirfoilTools as at
import pandas as pd
import os

if os.getlogin() == 'joeac':
    WDIR = "C:/WDIR/MoProMa_Auswertung/"
else:
    WDIR = "D:/Python_Codes/Workingdirectory_Auswertung"

file_path_messpkt = os.path.join(WDIR, 'Messpunkte Demonstrator_Mue13-33.xlsx')
file_path_messpkt_new = os.path.join(WDIR, 'Messpunkte Demonstrator_Mue13-33_new.xlsx')
file_path_airfoil = "C:/XFOIL6.99/mue13-33-le15.dat"

cols = ["Messpunkt Name", "Name Auswertung", "x", "x_norm"]

df = pd.read_excel(file_path_messpkt, names=cols, sheet_name="Tabelle2", usecols="B:E", skiprows=0)

foil = at.Airfoil(file_path_airfoil)



for i in df.index:
    if i > df.loc[:, "x_norm"].argmin():
        topside = False
    else:
        topside = True
    s = at.s_curve(at.usearch_x(df.loc[i, "x_norm"], foil.tck, topside=topside), foil.tck)
    df.loc[i, "s_norm"] = s

df.loc[:, "s"] = df.loc[:, "s_norm"] * 700

df.to_excel(file_path_messpkt_new, index=False)
print("done")
