# Pkg imports
import numpy as np

def determine_density(h):
    # Constants for the standard atmosphere model
    rho0 = 1.225   # [kg/m^3] Sea level air density
    h0 = 0         # [m]      Altitude at sea level
    H = 8500       # [m]      Scale height

    # Calculate air density using exponential decrease with altitude
    rho = rho0 * np.exp(-(h - h0) / H)

    # Return
    return rho
