# Pkg imports
import matplotlib.pyplot as plt

# Local imports
from utils import normalize_radians

class Controller:
    def __init__(self, Kp, Ki, Kd, setpoint, bounds):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.bounds = bounds
        self.prev_error = 0
        self.integral = 0

    def get_psi_correction(self, theta, psi_correction_prior, dt):

        # Pick ideal thrust vector angle
        if dt <= 0:
            psi_correction = psi_correction_prior
        else:
            error = self.setpoint - theta
            self.integral = self.integral + (error * dt)
            derivative = (error - self.prev_error)/dt
            psi_correction = (self.Kp * error + self.Ki * self.integral + self.Kd * derivative) 
            self.prev_error = error

        # # Bound the angle to something realistic
        lower_bound = theta - self.bounds
        upper_bound = theta  + self.bounds
        if psi_correction > upper_bound:
            psi_correction = upper_bound
        elif psi_correction < lower_bound:
            psi_correction = lower_bound

        # Return
        return normalize_radians(psi_correction)
