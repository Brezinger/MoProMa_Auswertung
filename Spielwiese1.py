import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
M = 0.0285  # kg/mol
R = 8.315  # J/(mol·K)
coeff = 1.46e-6  # coefficient for the new equation
l = 0.7  # m

# Temperature range in Kelvin
T = np.linspace(268.45, 308.15, 500)  # K

# Create a DataFrame
df = pd.DataFrame({'Temperature_K': T})

# Convert Temperature from Kelvin to Celsius
df['Temperature_C'] = df['Temperature_K'] - 273.15

# Calculate v in m/s using the new equation
df['v_m_per_s'] = (M * coeff * df['Temperature_K']**1.5) / (df['Temperature_K'] * R * l * (df['Temperature_K'] + 110.4))

# Convert v from m/s to km/h
df['v_km_per_h'] = df['v_m_per_s'] * 3.6

# Plot
plt.figure(figsize=(10, 8))
plt.plot(df['Temperature_C'], df['v_km_per_h'], label=r'$v = \frac{M \cdot 1.46 \cdot 10^{-6} \cdot T^{1.5}}{T \cdot R \cdot l \cdot (T+110.4)}$')
plt.xlabel('Temperature (°C)', fontsize=16)
plt.ylabel('v (km/h)', fontsize=16)
plt.title('Plot of $v$ vs Temperature', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
