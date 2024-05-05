# Pkg imports
import numpy as np

class Controller:
    def __init__(self, Kp, Ki, Kd, setpoint, bounds):
        self.Kp = Kp                    # [dimless]
        self.Ki = Ki                    # [dimless]
        self.Kd = Kd                    # [dimless]
        self.setpoint = setpoint        # [deg]
        self.bounds = bounds            # [deg]
        self.prev_error = 0             # [rad]
        self.integral = 0               # [rad]

    def get_psi_correction(self, theta, psi_correction_prior, error_prior, dt):

        # Units of inputs are:
        # theta                 [rads]
        # psi_correction_prior  [rads]
        # error_prior           [rads]

        # Swap theta to degrees (-180 to 180)
        theta = np.degrees(theta)               # [deg]
        if theta > 180:
            theta = theta - 360

        # Swap psi and error to degrees
        psi_correction_prior = np.degrees(psi_correction_prior)
        error_prior = np.degrees(error_prior)

        # Pick ideal thrust vector angle (in degrees)
        if dt <= 0:
            psi_correction = psi_correction_prior
            error = error_prior
        else:
            error = self.setpoint - theta
            self.integral = self.integral + (error * dt)
            derivative = (error - self.prev_error)/dt
            psi_correction = (self.Kp * error + self.Ki * self.integral + self.Kd * derivative) 
            self.prev_error = error

            # Bound the angle to something realistic (this is all in degrees)
            lower_bound = theta - self.bounds
            upper_bound = theta  + self.bounds
            if psi_correction > upper_bound:
                psi_correction = upper_bound
            elif psi_correction < lower_bound:
                psi_correction = lower_bound

        # Return (Swap error and psi_correction back to rads)
        return np.radians(psi_correction), np.radians(error)
