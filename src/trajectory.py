# Pkg imports
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Local imports
from controller import Controller
from util import normalize_radians

#------------------------------------------------
# Inputs
#------------------------------------------------
# Constants
Re = 6.3781e6

# Rocket Defintions
diam = 3.05
Ap = np.pi/(4*diam**2)
Cd = 0.2
Cl = 0.3

# Masses
mprop_ascent = 111130
mpl = 32000
mstruct = 6736
mprop_descent = mstruct*2
m_init = mprop_ascent + mstruct + mpl + mprop_descent

# Ascent Burn
tb_ascent = 100
mdot_ascent = mprop_ascent/tb_ascent
thrust_ascent = 19e5
h_turn = 1000

# Descent Burn
descent_ratio = 5
mdot_descent = mdot_ascent/descent_ratio
thrust_descent = thrust_ascent/descent_ratio

# Initial Conditions
v_init = 0
theta_init = np.radians(88)
h_init = 0
x_init = 0

#------------------------------------------------
# Controller
#------------------------------------------------
desired_flight_path_angle = 0
bounds = 60

#------------------------------------------------
# Dict
#------------------------------------------------
util_values = {
                'psi': theta_init,
                'theta_error': 0,
                'dt': 0,
                'prior_t': 0,
                'prior_h': 0,
                'prior_rho': 1.225,
                'prior_g': 9.81,
                'hmax_flag': False,
                'hmax': 0,
                'descent_t': 0,
                'mp_descent': mprop_descent,
                'mp_ascent': mprop_ascent,
                'phase': "None"
                }

save_values =  { 
                't': [],
                'psi': [],
                'theta_error': [],
                'theta': [],
                'v': [],
                'h': [],
                'x': [],
                'rho': [],
                'g': [],
                'F': [],
                'm': [],
                'mp_descent': [],
                'mp_ascent': [],
                'phase': []
                }

def determine_density(h):
    # Constants for the standard atmosphere model
    rho0 = 1.225  # Sea level air density in kg/m^3
    h0 = 0        # Altitude at sea level in meters
    H = 8500      # Scale height in meters

    # Calculate air density using exponential decrease with altitude
    rho = rho0 * np.exp(-(h - h0) / H)

    return rho

def ode_equations(t,y):

    # Set parameters of interest
    v = y[0]
    theta = y[1]
    h = y[2]
    x = y[3]

    # Init
    global util_values
    global save_values
    
    # Determine density and gravity as a fcn of height
    if h > 0:
        rho = determine_density(h)
        g = 9.81 * (Re/(Re + h))**2
    else:
        rho = util_values["prior_rho"]
        g = util_values["prior_g"]

    # Determine dt
    util_values['dt'] = t - util_values['prior_t']

    # Normalize theta
    theta = normalize_radians(theta)

    # Ascent: before gravity turn
    if h <= h_turn and t<=tb_ascent:

        # Phase
        util_values["phase"] = 1
        Cl = 0

        # Determine TVC correction
        util_values['psi'], util_values['theta_error'] =pidController.get_psi_correction(theta, util_values['psi'], util_values['theta_error'], util_values['dt'])
        util_values['psi'] = theta

        # Thrust and mass
        m = m_init - (mdot_ascent * t)
        util_values['mp_ascent'] = util_values['mp_ascent'] - (mdot_ascent * util_values['dt'])
        F = thrust_ascent

        # EOMs
        v_dot = (F*np.cos(util_values['psi']-theta))/m - (Cd*rho*v**2*Ap)/(2*m) - g*np.sin(theta)
        theta_dot = 0
        h_dot = v*np.sin(theta)
        x_dot = v*np.cos(theta)
        
    # Ascent burn
    elif t <= tb_ascent:

        # Phase
        util_values["phase"] = 2
        Cl = 0
     
        # Determine TVC correction
        util_values['psi'], util_values['theta_error'] =pidController.get_psi_correction(theta, util_values['psi'], util_values['theta_error'], util_values['dt'])
        util_values['psi'] = theta

        # Thrust and mass
        m = m_init - (mdot_ascent * t)
        util_values['mp_ascent'] = util_values['mp_ascent'] - (mdot_ascent * util_values['dt'])
        F = thrust_ascent

        # EOMS
        v_dot = (F*np.cos(util_values['psi']-theta))/m - (Cd*rho*v**2*Ap)/(2*m) - g*np.sin(theta)
        theta_dot = (F*np.sin(util_values['psi']-theta))/(m*v) + (Cl*rho*v*Ap)/(2*m) - (g*np.cos(theta))/v
        h_dot = v*np.sin(theta)
        x_dot = v*np.cos(theta)

    # Coast phase
    elif t > tb_ascent and util_values["hmax_flag"] == False:

        # Phase
        util_values["phase"] = 3
        Cl = 0

        # Determine TVC correction
        util_values['psi'], util_values['theta_error'] =pidController.get_psi_correction(theta, util_values['psi'], util_values['theta_error'], util_values['dt'])
        util_values['psi'] = theta

        # Thrust and mass
        m = m_init - (mdot_ascent * tb_ascent)
        F = 0

        # EOMS
        v_dot = (F*np.cos(util_values['psi']-theta))/m - (Cd*rho*v**2*Ap)/(2*m) - g*np.sin(theta)
        theta_dot = (F*np.sin(util_values['psi']-theta))/(m*v) + (Cl*rho*v*Ap)/(2*m) - (g*np.cos(theta))/v
        h_dot = v*np.sin(theta)
        x_dot = v*np.cos(theta)

    # Descent burn
    elif util_values["hmax_flag"] and util_values['mp_descent'] > 0 and util_values['hmax'] - h >= 10e3:

        # Phase
        util_values["phase"] = 4
        Cl = 0
        
        # Determine TVC correction
        util_values['psi'], util_values['theta_error'] = pidController.get_psi_correction(theta, util_values['psi'], util_values['theta_error'], util_values['dt'])

        # Set times
        util_values['descent_t'] = util_values['descent_t'] + util_values['dt']

        # Thrust and mass
        util_values['mp_descent'] = util_values['mp_descent'] - (mdot_descent *  util_values['dt'])
        m = m_init - (mdot_ascent * tb_ascent) - (mdot_descent * util_values['descent_t']) - mpl
        F_v = 0
        F_theta = thrust_descent
        F = F_theta

        # EOMS
        v_dot = (F_v*np.cos(util_values['psi']-theta))/m - (Cd*rho*v**2*Ap)/(2*m) - g*np.sin(theta)
        theta_dot = ((F_theta*np.sin(util_values['psi']-theta))/(m*v) + (Cl*rho*v*Ap)/(2*m) - (g*np.cos(theta))/v)
        h_dot = v*np.sin(theta)
        x_dot = v*np.cos(theta)

    # Post descent
    else:

        # Phase
        util_values["phase"] = 5
        Cl = 0

        # Determine TVC correction
        util_values['psi'], util_values['theta_error'] =pidController.get_psi_correction(theta, util_values['psi'], util_values['theta_error'], util_values['dt'])
        util_values['psi'] = theta

        # Thrust and mass
        m = m_init - (mdot_ascent * tb_ascent) - (mdot_descent * util_values['descent_t']) - mpl
        F = 0

        # EOMS
        v_dot = (F*np.cos(util_values['psi']-theta))/m - (Cd*rho*v**2*Ap)/(2*m) - g*np.sin(theta)
        theta_dot = (F*np.sin(util_values['psi']-theta))/(m*v) + (Cl*rho*v*Ap)/(2*m) - (g*np.cos(theta))/v
        h_dot = v*np.sin(theta)
        x_dot = v*np.cos(theta)

    # Determine if we have reached maximum height
    if (h - util_values['prior_h']) < -10 and util_values["hmax_flag"] == False:
        util_values['hmax_flag'] = True 
        util_values['hmax'] = util_values['prior_h']

    # Save values
    save_values['t'].append(t)
    save_values['F'].append(F)
    save_values['m'].append(m)
    save_values['mp_ascent'].append(util_values['mp_ascent'])
    save_values['mp_descent'].append(util_values['mp_descent'])
    save_values['psi'].append(util_values['psi'])
    save_values['theta_error'].append(util_values['theta_error'])
    save_values['theta'].append(theta)
    save_values['v'].append(v)
    save_values['h'].append(h)
    save_values['x'].append(x)
    save_values['rho'].append(rho)
    save_values['g'].append(g)
    save_values['phase'].append(util_values['phase'])

    # Update prior values
    util_values['prior_t'] = t
    util_values['prior_h'] = h
    util_values['prior_rho'] = rho
    util_values['prior_g'] = g
    
    #print(t, util_values["phase"], theta*180/np.pi,util_values['theta_error']*180/np.pi, util_values['mp_descent'])

    # Return
    return [v_dot, theta_dot, h_dot, x_dot]

def generate_trajectory(pidController):

    #------------------------------------------------
    # ODE Solving
    #------------------------------------------------
    # Solve ODEs
    sol = integrate.solve_ivp(ode_equations, (0,5000), [v_init, theta_init, h_init, x_init], max_step = 0.1)

    # Final values
    v = sol.y[0]                        # m/s
    theta = sol.y[1]                    # radians
    h = sol.y[2]                        # m
    x = sol.y[3]                        # m
    t = sol.t                           # s
   
    #------------------------------------------------
    # Post Processing
    #------------------------------------------------
    # Convert to degrees
    save_values['psi'] = np.degrees(save_values['psi'])
    save_values['theta_error'] = np.degrees(save_values['theta_error'])
    save_values['theta'] = np.degrees(save_values['theta'])

    # Convert to km/s, km, and Metric Tons, and kN
    for i in range(0,len(save_values['t'])):
        save_values['x'][i] = save_values['x'][i]/1e3                       # [km]
        save_values['h'][i] = save_values['h'][i]/1e3                       # [km]
        save_values['v'][i] = save_values['v'][i]/1e3                       # [km/s]
        save_values['F'][i] = save_values['F'][i]/1e3                       # [kN]
        save_values['m'][i] = save_values['m'][i]/1e3                       # [MT]
        save_values['mp_descent'][i] = save_values['mp_descent'][i]/1e3     # [MT]
        save_values['mp_ascent'][i] = save_values['mp_ascent'][i]/1e3       # [MT]

    # Post processing
    h_array = np.array(save_values['h'])
    reversed_h_array = h_array[::-1]
    last_index_positive = len(h_array) - np.argmax(reversed_h_array > 0) - 1

    #------------------------------------------------
    # Subplot
    #------------------------------------------------
    # # Create subplots with 4 rows and 1 column
    # fig, axs = plt.subplots(9, 1, figsize=(10, 32))
    # fig_num = -1

    # # Plot Phase
    # fig_num = fig_num + 1
    # axs[fig_num].plot(save_values['t'], save_values['phase'])
    # axs[fig_num].set_xlabel('Time (s)')
    # axs[fig_num].set_ylabel('Phase')
    # axs[fig_num].set_title('Phase')
    # axs[fig_num].grid(True)
    # axs[fig_num].set_xlim(0, save_values['t'][last_index_positive])

    # # Plot Height
    # fig_num = fig_num + 1
    # axs[fig_num].plot(save_values['t'], save_values['h'])
    # axs[fig_num].set_xlabel('Time (s)')
    # axs[fig_num].set_ylabel('Height (km)')
    # axs[fig_num].set_title('Height')
    # axs[fig_num].grid(True)
    # axs[fig_num].set_xlim(0, save_values['t'][last_index_positive])
    # axs[fig_num].set_ylim(0, max(save_values['h'])*1.1)

    # # Plot Downrange
    # fig_num = fig_num + 1
    # axs[fig_num].plot(save_values['t'], save_values['x'])
    # axs[fig_num].set_xlabel('Time (s)')
    # axs[fig_num].set_ylabel('Distance (km)')
    # axs[fig_num].set_title('Downrange')
    # axs[fig_num].grid(True)
    # axs[fig_num].set_xlim(0, save_values['t'][last_index_positive])
    # axs[fig_num].set_ylim(0, max(save_values['x'][0:last_index_positive])*1.1)

    # # Plot Velocity
    # fig_num = fig_num + 1    
    # axs[fig_num].plot(save_values['t'], save_values['v'])
    # axs[fig_num].set_xlabel('Time (s)')
    # axs[fig_num].set_ylabel('Velocity (km/s)')
    # axs[fig_num].set_title('Velocity')
    # axs[fig_num].grid(True)
    # axs[fig_num].set_xlim(0, save_values['t'][last_index_positive])
    # axs[fig_num].set_ylim(0, max(save_values['v'][0:last_index_positive])*1.1)

    # # Plot Angles
    # fig_num = fig_num + 1
    # axs[fig_num].plot(save_values['t'], save_values['theta'], label="$\\theta$")
    # axs[fig_num].plot(save_values['t'], save_values['theta_error'], label="$\\theta_e$")
    # axs[fig_num].plot(save_values['t'], save_values['psi'], label="$\\psi$")
    # axs[fig_num].set_xlabel('Time (s)')
    # axs[fig_num].set_ylabel('Angle (deg)')
    # axs[fig_num].set_title('Flight Angles')
    # axs[fig_num].grid(True)
    # axs[fig_num].legend()
    # axs[fig_num].set_xlim(0, save_values['t'][last_index_positive])

    # # Mass
    # fig_num = fig_num + 1
    # axs[fig_num].plot(save_values['t'], save_values['m'], label="Total Mass")
    # axs[fig_num].plot(save_values['t'], save_values['mp_descent'], label="Descent Propellant Mass")
    # axs[fig_num].plot(save_values['t'], save_values['mp_ascent'], label="Ascent Propellant Mass")
    # axs[fig_num].set_xlabel('Time (s)')
    # axs[fig_num].set_ylabel('Mass [MT]')
    # axs[fig_num].set_title('Mass')
    # axs[fig_num].grid(True)
    # axs[fig_num].legend()
    # axs[fig_num].set_xlim(0, save_values['t'][last_index_positive])

    # # Thrust
    # fig_num = fig_num + 1
    # axs[fig_num].plot(save_values['t'], save_values['F'], label="Thrust")
    # axs[fig_num].set_xlabel('Time (s)')
    # axs[fig_num].set_ylabel('Thrust (kN)')
    # axs[fig_num].set_title('Thrust')
    # axs[fig_num].grid(True)
    # axs[fig_num].set_xlim(0, save_values['t'][last_index_positive])

    # # Density
    # fig_num = fig_num + 1
    # axs[fig_num].plot(save_values['t'], save_values['rho'], label="Density")
    # axs[fig_num].set_xlabel('Time (s)')
    # axs[fig_num].set_ylabel('Density (kg/m$^3$)')
    # axs[fig_num].set_title('Density')
    # axs[fig_num].grid(True)
    # axs[fig_num].set_xlim(0, save_values['t'][last_index_positive])

    # # Gravity
    # fig_num = fig_num + 1
    # axs[fig_num].plot(save_values['t'], save_values['g'], label="Gravity")
    # axs[fig_num].set_xlabel('Time (s)')
    # axs[fig_num].set_ylabel('Gravity (m/s$^2$)')
    # axs[fig_num].set_title('Gravity')
    # axs[fig_num].grid(True)
    # axs[fig_num].set_xlim(0, save_values['t'][last_index_positive])

    # plt.tight_layout()
    # plt.savefig('data/output/images/combined_plots.png')

    # #------------------------------------------------
    # # Trajectory Plot
    # #------------------------------------------------
    # # Create a figure and axes
    # fig, ax = plt.subplots(figsize=(10,8))
    # ax.plot(save_values['x'], save_values['h'], color='gray', linestyle="--")
    # for i in range(0, len(save_values['t']), 500):
    #     flight_angle_vector_x = np.cos(save_values['theta'][i] * np.pi / 180)
    #     flight_angle_vector_y = np.sin(save_values['theta'][i] * np.pi / 180)
    #     thrust_angle_vector_x = np.cos(save_values['psi'][i] * np.pi / 180)
    #     thrust_angle_vector_y = np.sin(save_values['psi'][i] * np.pi / 180)
    #     ax.quiver(save_values['x'][i], save_values['h'][i], flight_angle_vector_x, flight_angle_vector_y, scale=25, color='r', label='Body Vector')
    #     ax.quiver(save_values['x'][i], save_values['h'][i], -thrust_angle_vector_x, -thrust_angle_vector_y, scale=35, color='b', label='Thrust Vector')
    # ax.set_xlabel('Downrange Position (km)')
    # ax.set_ylabel('Height (km)')
    # ax.set_title('2D Position')
    # ax.grid(True)
    # ax.legend(handles=ax.get_legend_handles_labels()[0][:2], labels=ax.get_legend_handles_labels()[1][:2])          # This takes only the first two entries of the legend because it repeats
    # ax.set_xlim(0, max(save_values['x'][0:last_index_positive]))
    # ax.set_ylim(0, save_values['h'][np.argmax(save_values['h'])]*1.1)

    # # Save the plot
    # plt.savefig('data/output/images/combined_trajectory.png')

    # #------------------------------------------------
    # # Trajectory Animation
    # #------------------------------------------------
    # # Create a figure and axes
    # fig, ax = plt.subplots(figsize=(10, 8))

    # # Create quiver plots outside the loop
    # flight_arrow = ax.quiver([], [], [], [], scale=25, color='r', label='Body Vector')
    # thrust_arrow = ax.quiver([], [], [], [], scale=35, color='b', label='Thrust Vector')

    # # Function to update the plot for each frame of animation
    # def update(frame):
    #     i = frame * 500
    #     x_pos = save_values['x'][i]
    #     y_pos = save_values['h'][i]
    #     flight_angle_vector_x = np.cos(save_values['theta'][i] * np.pi / 180)
    #     flight_angle_vector_y = np.sin(save_values['theta'][i] * np.pi / 180)
    #     thrust_angle_vector_x = np.cos(save_values['psi'][i] * np.pi / 180)
    #     thrust_angle_vector_y = np.sin(save_values['psi'][i] * np.pi / 180)
    #     flight_arrow.set_offsets((x_pos, y_pos))
    #     flight_arrow.set_UVC(flight_angle_vector_x, flight_angle_vector_y)
    #     thrust_arrow.set_offsets((x_pos, y_pos))
    #     thrust_arrow.set_UVC(-thrust_angle_vector_x, -thrust_angle_vector_y)
    #     return flight_arrow, thrust_arrow

    # # Call update_quiver function inside the loop
    # ani = FuncAnimation(fig, update, frames=len(save_values['t']) // 500, repeat=True) # interval?
    
    # # Set axes labels and title
    # ax.set_xlabel('Downrange Position (km)')
    # ax.set_ylabel('Height (km)')
    # ax.set_title('2D Position')
    # ax.grid(True)
    # ax.legend()
    # ax.set_xlim(0, max(save_values['x'][0:last_index_positive]))
    # ax.set_ylim(0, save_values['h'][np.argmax(save_values['h'])]*1.1)
    # ani.save('data/output/images/combined_trajectory.gif', writer='pillow')

    #------------------------------------------------
    # Printing
    #------------------------------------------------
    #print("Max Height [km]:",util_values['hmax']/1e3)
    print(f"Run Complete for: Kp={pidController.Kp}, Ki={pidController.Ki}, Kd={pidController.Kd}")

    # Return
    return save_values, last_index_positive

if __name__ == '__main__':

    # Init
    values_list = []
    max_down_range_list = []
    max_height_list = []

    # PID
    gain_list = [[100, 1, 1], [10, 0, 0], [15, 1, 6], [4, 1, 36], [150, 0, 6], [0, 0, 6], [150, 100, 60]]
    for gains in gain_list:

        pidController = Controller(gains[0], gains[1], gains[2], desired_flight_path_angle, bounds)

        # Call trajs
        values, last_index_positive = generate_trajectory(pidController)
        values_list.append(values.copy())
        max_down_range_list.append(values['x'][last_index_positive])
        max_height_list.append(max(values['h'][0:last_index_positive]))

        # Clear values
        for key in save_values:
            save_values[key] = []

        # Reinit
        util_values = {
                'psi': theta_init,
                'theta_error': 0,
                'dt': 0,
                'prior_t': 0,
                'prior_h': 0,
                'prior_rho': 1.225,
                'prior_g': 9.81,
                'hmax_flag': False,
                'hmax': 0,
                'descent_t': 0,
                'mp_descent': mprop_descent,
                'mp_ascent': mprop_ascent,
                'phase': "None"
                }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for values, gains in zip(values_list, gain_list):
        ax.plot(values['x'], values['h'], label=f"K$_p$={gains[0]}, K$_i$={gains[1]}, K$_d$={gains[2]}")
    ax.set_xlabel('Downrage (km)')
    ax.set_ylabel('Height (km)')
    ax.set_title('2D Position')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, max(max_down_range_list))
    ax.set_ylim(0, max(max_height_list)*1.1)
    plt.savefig('data/output/images/dispersed_trajectories.png')




        