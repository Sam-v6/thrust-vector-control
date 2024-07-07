# Pkg imports
import numpy as np

# Local imports
from util import normalize_radians
from util import load_input_data
from environment import determine_density


class OdeSolver:
    def __init__(self, pidController, rocket_params, initial_conditions):

        # Objects and constants
        self.controller = pidController
        self.rocket_params, self.initial_conditions = rocket_params, initial_conditions

        # ODE parameters of interest
        self.v = 0
        self.theta = 0
        self.h = 0
        self.x = 0

        # Parameters to save
        self.v_history = []
        self.theta_history = []
        self.h_history = []
        self.x_history = []

        # Utility values
        self.i = 0
        self.hmax_flag = False
        self.hmax = 0
        self.descent_t = 0
        self.prior_t = 0
        self.prior_h = 0
        self.prior_rho =  1.225
        self.prior_g = self.rocket_params["GRAVITY_CONSTANT"]
        self.theta_error = 0
        self.dt = 0
        self.m = 0
        self.F = 0
        self.v_dot = 0
        self.theta_dot = 0
        self.h_dot = 0
        self.x_dot = 0
        self.rho = 0
        self.g = self.rocket_params["MASS_PROP_DESCENT"]
        self.mp_descent = self.rocket_params["MASS_PROP_DESCENT"]
        self.mp_ascent = self.rocket_params["MASS_PROP_ASCENT"]
        self.psi = 0

        # Intermediate values to save
        self.t_history = []
        self.F_history = []
        self.m_history = []
        self.mp_ascent_history = []
        self.mp_descent_history = []
        self.psi_history = []
        self.theta_error_history = []
        self.rho_history = []
        self.g_history = []
        self.phase_history = []
        self.g_history = []

    def calculate_trajectory(self, t, y):

        # Set parameters of interest
        self.v = y[0]
        self.theta = y[1]
        self.h = y[2]
        self.x = y[3]
        
        # Determine density and gravity as a fcn of height
        if self.h > 0:
            self.rho = determine_density(self.h)
            self.g = self.rocket_params["GRAVITY_CONSTANT"] * (self.rocket_params["RADIUS_EARTH"]/(self.rocket_params["RADIUS_EARTH"] + self.h))**2
        else:
            self.rho = self.prior_rho
            self.g = self.prior_g

        # Determine dt
        self.dt = t - self.prior_t

        # Normalize self.theta
        self.theta = normalize_radians(self.theta)

        # Ascent: before gravity turn
        if self.h <= self.rocket_params["HEIGHT_TURN"] and t<=self.rocket_params["TIME_ASCENT_BURN"]:

            # Phase
            self.phase = 1
            LIFT_CF = 0

            # Determine TVC correction
            self.psi, self.theta_error = self.controller.get_psi_correction(self.theta, self.psi, self.theta_error, self.dt)
            self.psi = self.theta

            # Thrust and mass
            self.m = self.rocket_params["MASS_INIT"] - (self.rocket_params["MDOT_ASCENT"] * t)
            self.mp_ascent = self.mp_ascent - (self.rocket_params["MDOT_ASCENT"] * self.dt)
            self.F = self.rocket_params["THRUST_ASCENT"]

            # EOMs
            self.v_dot = (self.F*np.cos(self.psi-self.theta))/self.m - (self.rocket_params["DRAG_CF"]*self.rho*self.v**2*self.rocket_params["PROFILE_AREA"])/(2*self.m) - self.g*np.sin(self.theta)
            self.theta_dot = 0
            self.h_dot = self.v*np.sin(self.theta)
            self.x_dot = self.v*np.cos(self.theta)
            
        # Ascent burn
        elif t <= self.rocket_params["TIME_ASCENT_BURN"]:

            # Phase
            self.phase = 2
            LIFT_CF = 0
        
            # Determine TVC correction
            self.psi, self.theta_error = self.controller.get_psi_correction(self.theta, self.psi, self.theta_error, self.dt)
            self.psi = self.theta

            # Thrust and mass
            self.m = self.rocket_params["MASS_INIT"] - (self.rocket_params["MDOT_ASCENT"] * t)
            self.mp_ascent = self.mp_ascent - (self.rocket_params["MDOT_ASCENT"] * self.dt)
            self.F = self.rocket_params["THRUST_ASCENT"]

            # EOMS
            self.v_dot = (self.F*np.cos(self.psi-self.theta))/self.m - (self.rocket_params["DRAG_CF"]*self.rho*self.v**2*self.rocket_params["PROFILE_AREA"])/(2*self.m) - self.g*np.sin(self.theta)
            self.theta_dot = (self.F*np.sin(self.psi-self.theta))/(self.m*self.v) + (LIFT_CF*self.rho*self.v*self.rocket_params["PROFILE_AREA"])/(2*self.m) - (self.g*np.cos(self.theta))/self.v
            self.h_dot = self.v*np.sin(self.theta)
            self.x_dot = self.v*np.cos(self.theta)

        # Coast phase
        elif t > self.rocket_params["TIME_ASCENT_BURN"] and self.hmax_flag == False:

            # Phase
            self.phase = 3
            LIFT_CF = 0

            # Determine TVC correction
            self.psi, self.theta_error =self.controller.get_psi_correction(self.theta, self.psi, self.theta_error, self.dt)
            self.psi = self.theta

            # Thrust and mass
            self.m = self.rocket_params["MASS_INIT"] - (self.rocket_params["MDOT_ASCENT"] * self.rocket_params["TIME_ASCENT_BURN"])
            self.F = 0

            # EOMS
            self.v_dot = (self.F*np.cos(self.psi-self.theta))/self.m - (self.rocket_params["DRAG_CF"]*self.rho*self.v**2*self.rocket_params["PROFILE_AREA"])/(2*self.m) - self.g*np.sin(self.theta)
            self.theta_dot = (self.F*np.sin(self.psi-self.theta))/(self.m*self.v) + (LIFT_CF*self.rho*self.v*self.rocket_params["PROFILE_AREA"])/(2*self.m) - (self.g*np.cos(self.theta))/self.v
            self.h_dot = self.v*np.sin(self.theta)
            self.x_dot = self.v*np.cos(self.theta)

        # Descent burn
        elif self.hmax_flag and self.mp_descent > 0 and self.hmax - self.h >= 10e3:

            # Phase
            self.phase = 4
            LIFT_CF = 0
            
            # Determine TVC correction
            self.psi, self.theta_error = self.controller.get_psi_correction(self.theta, self.psi, self.theta_error, self.dt)

            # Set times
            self.descent_t = self.descent_t + self.dt

            # Thrust and mass
            self.mp_descent = self.mp_descent - (self.rocket_params["MDOT_DESCENT"] *  self.dt)
            self.m = self.rocket_params["MASS_INIT"] - (self.rocket_params["MDOT_ASCENT"] * self.rocket_params["TIME_ASCENT_BURN"]) - (self.rocket_params["MDOT_DESCENT"] * self.descent_t) - self.rocket_params["MASS_PAYLOAD"]
            F_v = 0
            F_theta = self.rocket_params["THRUST_DESCENT"]
            self.F = F_theta

            # EOMS
            self.v_dot = (F_v*np.cos(self.psi-self.theta))/self.m - (self.rocket_params["DRAG_CF"]*self.rho*self.v**2*self.rocket_params["PROFILE_AREA"])/(2*self.m) - self.g*np.sin(self.theta)
            self.theta_dot = ((F_theta*np.sin(self.psi-self.theta))/(self.m*self.v) + (LIFT_CF*self.rho*self.v*self.rocket_params["PROFILE_AREA"])/(2*self.m) - (self.g*np.cos(self.theta))/self.v)
            self.h_dot = self.v*np.sin(self.theta)
            self.x_dot = self.v*np.cos(self.theta)

        # Post descent
        else:

            # Phase
            self.phase = 5
            LIFT_CF = 0

            # Determine TVC correction
            self.psi, self.theta_error = self.controller.get_psi_correction(self.theta, self.psi, self.theta_error, self.dt)
            self.psi = self.theta

            # Thrust and mass
            self.m = self.rocket_params["MASS_INIT"] - (self.rocket_params["MDOT_ASCENT"] * self.rocket_params["TIME_ASCENT_BURN"]) - (self.rocket_params["MDOT_DESCENT"] * self.descent_t) - self.rocket_params["MASS_PAYLOAD"]
            self.F = 0

            # EOMS
            self.v_dot = (self.F*np.cos(self.psi-self.theta))/self.m - (self.rocket_params["DRAG_CF"]*self.rho*self.v**2*self.rocket_params["PROFILE_AREA"])/(2*self.m) - self.g*np.sin(self.theta)
            self.theta_dot = (self.F*np.sin(self.psi-self.theta))/(self.m*self.v) + (LIFT_CF*self.rho*self.v*self.rocket_params["PROFILE_AREA"])/(2*self.m) - (self.g*np.cos(self.theta))/self.v
            self.h_dot = self.v*np.sin(self.theta)
            self.x_dot = self.v*np.cos(self.theta)

        # Determine if we have reached maximum height
        if (self.h - self.prior_h) < -10 and self.hmax_flag == False:
            self.hmax_flag = True 
            self.hmax = self.prior_h

        # Save values
        self.t_history.append(t)
        self.F_history.append(self.F)
        self.m_history.append(self.m)
        self.mp_ascent_history.append(self.mp_ascent)
        self.mp_descent_history.append(self.mp_descent)
        self.psi_history.append(self.psi)
        self.theta_error_history.append(self.theta_error)
        self.theta_history.append(self.theta)
        self.v_history.append(self.v)
        self.h_history.append(self.h)
        self.x_history.append(self.x)
        self.rho_history.append(self.rho)
        self.g_history.append(self.g)
        self.phase_history.append(self.phase)

        # Update prior values
        self.prior_t = t
        self.prior_h = self.h
        self.prior_rho = self.rho
        self.prior_g = self.g
        
        # Return
        return [self.v_dot, self.theta_dot, self.h_dot, self.x_dot]