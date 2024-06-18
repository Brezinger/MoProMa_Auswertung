import numpy as np

# Example data
time_intervals = np.arange(0, 100, 1)  # time intervals from 0 to 99
variances = np.random.normal(loc=0.5, scale=0.1, size=100)  # example variances

# Function to determine equilibrium time
def find_equilibrium(time_intervals, variances, threshold=0.01, consecutive=5):
    for i in range(consecutive, len(variances)):
        if all(abs((variances[j] - variances[j-1]) / variances[j-1]) < threshold for j in range(i, i+consecutive)):
            return time_intervals[i]
    return None  # equilibrium not reached within the given data

equilibrium_time = find_equilibrium(time_intervals, variances)
print("Equilibrium time:", equilibrium_time)