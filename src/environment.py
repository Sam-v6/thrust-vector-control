# Pkg imports
import numpy as np

def determine_density(h):
    # Constants for the standard atmosphere model
    rho0 = 1.225  # Sea level air density in kg/m^3
    h0 = 0        # Altitude at sea level in meters
    H = 8500      # Scale height in meters

    # Calculate air density using exponential decrease with altitude
    rho = rho0 * np.exp(-(h - h0) / H)

    # Return
    return rho
