# Pkg imports
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Local imports
from controller import Controller
from ode_solver import OdeSolver
from util import load_input_data

def generate_individual_trajectory(pidController, rocket_params, initial_conditions):

    # Create ODE object
    odeSolver = OdeSolver(pidController, rocket_params, initial_conditions)

    # Solve ODEs
    sol = integrate.solve_ivp(odeSolver.calculate_trajectory, (0,5000), [initial_conditions["v_init"], initial_conditions["theta_init"], initial_conditions["h_init"], initial_conditions["x_init"]], max_step = 0.1)

    # Grab final values
    v = sol.y[0]                        # m/s
    theta = sol.y[1]                    # radians
    h = sol.y[2]                        # m
    x = sol.y[3]                        # m
    t = sol.t                           # s

    # Return
    return odeSolver

def plot_individual_trajectory(odeSolver):

    #------------------------------------------------
    # Post Processing
    #------------------------------------------------
    # Convert to degrees
    odeSolver.psi_history = np.degrees(odeSolver.psi_history)
    odeSolver.theta_error_history = np.degrees(odeSolver.theta_error_history)
    odeSolver.theta_history = np.degrees(odeSolver.theta_history)

    # Convert to km/s, km, and Metric Tons, and kN
    for i in range(0,len(odeSolver.t_history)):
        odeSolver.x_history[i] = odeSolver.x_history[i]/1e3                       # [km]
        odeSolver.h_history[i] = odeSolver.h_history[i]/1e3                       # [km]
        odeSolver.v_history[i] =odeSolver.v_history[i]/1e3                        # [km/s]
        odeSolver.F_history[i] = odeSolver.F_history[i]/1e3                       # [kN]
        odeSolver.m_history[i] = odeSolver.m_history[i]/1e3                       # [MT]
        odeSolver.mp_descent_history[i] = odeSolver.mp_descent_history[i]/1e3     # [MT]
        odeSolver.mp_ascent_history[i] = odeSolver.mp_ascent_history[i]/1e3       # [MT]

    # Post processing
    h_array = np.array(odeSolver.h_history)
    reversed_h_array = h_array[::-1]
    last_index_positive = len(h_array) - np.argmax(reversed_h_array > 0) - 1

    #------------------------------------------------
    # Subplot
    #------------------------------------------------
    # Create subplots with 4 rows and 1 column
    fig, axs = plt.subplots(3, 3, figsize=(35, 20))
    fig_num = -1

    # Plot Phase
    fig_num = fig_num + 1
    axs[0,0].plot(odeSolver.t_history, odeSolver.phase_history)
    axs[0,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Phase')
    axs[0,0].set_title('Phase')
    axs[0,0].grid(True)
    axs[0,0].set_xlim(0, odeSolver.t_history[last_index_positive])

    # Plot Height
    fig_num = fig_num + 1
    axs[1,0].plot(odeSolver.t_history, odeSolver.h_history)
    axs[1,0].set_xlabel('Time (s)')
    axs[1,0].set_ylabel('Height (km)')
    axs[1,0].set_title('Height')
    axs[1,0].grid(True)
    axs[1,0].set_xlim(0, odeSolver.t_history[last_index_positive])
    axs[1,0].set_ylim(0, max(odeSolver.h_history)*1.1)

    # Plot Downrange
    fig_num = fig_num + 1
    axs[2,0].plot(odeSolver.t_history, odeSolver.x_history)
    axs[2,0].set_xlabel('Time (s)')
    axs[2,0].set_ylabel('Distance (km)')
    axs[2,0].set_title('Downrange')
    axs[2,0].grid(True)
    axs[2,0].set_xlim(0, odeSolver.t_history[last_index_positive])
    axs[2,0].set_ylim(0, max(odeSolver.x_history[0:last_index_positive])*1.1)

    # Plot Velocity
    fig_num = fig_num + 1    
    axs[0,1].plot(odeSolver.t_history,odeSolver.v_history)
    axs[0,1].set_xlabel('Time (s)')
    axs[0,1].set_ylabel('Velocity (km/s)')
    axs[0,1].set_title('Velocity')
    axs[0,1].grid(True)
    axs[0,1].set_xlim(0, odeSolver.t_history[last_index_positive])
    axs[0,1].set_ylim(0, max(odeSolver.v_history[0:last_index_positive])*1.1)

    # Plot Angles
    fig_num = fig_num + 1
    axs[1,1].plot(odeSolver.t_history, odeSolver.theta_history, label="$\\theta$")
    axs[1,1].plot(odeSolver.t_history, odeSolver.theta_error_history, label="$\\theta_e$")
    axs[1,1].plot(odeSolver.t_history, odeSolver.psi_history, label="$\\psi$")
    axs[1,1].set_xlabel('Time (s)')
    axs[1,1].set_ylabel('Angle (deg)')
    axs[1,1].set_title('Flight Angles')
    axs[1,1].grid(True)
    axs[1,1].legend()
    axs[1,1].set_xlim(0, odeSolver.t_history[last_index_positive])

    # Mass
    fig_num = fig_num + 1
    axs[2,1].plot(odeSolver.t_history, odeSolver.m_history, label="Total Mass")
    axs[2,1].plot(odeSolver.t_history, odeSolver.mp_descent_history, label="Descent Propellant Mass")
    axs[2,1].plot(odeSolver.t_history, odeSolver.mp_ascent_history, label="Ascent Propellant Mass")
    axs[2,1].set_xlabel('Time (s)')
    axs[2,1].set_ylabel('Mass [MT]')
    axs[2,1].set_title('Mass')
    axs[2,1].grid(True)
    axs[2,1].legend()
    axs[2,1].set_xlim(0, odeSolver.t_history[last_index_positive])

    # Thrust
    fig_num = fig_num + 1
    axs[0,2].plot(odeSolver.t_history, odeSolver.F_history, label="Thrust")
    axs[0,2].set_xlabel('Time (s)')
    axs[0,2].set_ylabel('Thrust (kN)')
    axs[0,2].set_title('Thrust')
    axs[0,2].grid(True)
    axs[0,2].set_xlim(0, odeSolver.t_history[last_index_positive])

    # Density
    fig_num = fig_num + 1
    axs[1,2].plot(odeSolver.t_history, odeSolver.rho_history, label="Density")
    axs[1,2].set_xlabel('Time (s)')
    axs[1,2].set_ylabel('Density (kg/m$^3$)')
    axs[1,2].set_title('Density')
    axs[1,2].grid(True)
    axs[1,2].set_xlim(0, odeSolver.t_history[last_index_positive])

    # Gravity
    fig_num = fig_num + 1
    axs[2,2].plot(odeSolver.t_history, odeSolver.g_history, label="Gravity")
    axs[2,2].set_xlabel('Time (s)')
    axs[2,2].set_ylabel('Gravity (m/s$^2$)')
    axs[2,2].set_title('Gravity')
    axs[2,2].grid(True)
    axs[2,2].set_xlim(0, odeSolver.t_history[last_index_positive])

    plt.tight_layout()
    plt.savefig('data/output/combined_plots.png')

    #------------------------------------------------
    # Trajectory Plot
    #------------------------------------------------
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(odeSolver.x_history, odeSolver.h_history, color='gray', linestyle="--")
    for i in range(0, len(odeSolver.t_history), 500):
        flight_angle_vector_x = np.cos(odeSolver.theta_history[i] * np.pi / 180)
        flight_angle_vector_y = np.sin(odeSolver.theta_history[i] * np.pi / 180)
        thrust_angle_vector_x = np.cos(odeSolver.psi_history[i] * np.pi / 180)
        thrust_angle_vector_y = np.sin(odeSolver.psi_history[i] * np.pi / 180)
        ax.quiver(odeSolver.x_history[i], odeSolver.h_history[i], flight_angle_vector_x, flight_angle_vector_y, scale=25, color='r', label='Body Vector')
        ax.quiver(odeSolver.x_history[i], odeSolver.h_history[i], -thrust_angle_vector_x, -thrust_angle_vector_y, scale=35, color='b', label='Thrust Vector')
    ax.set_xlabel('Downrange Position (km)')
    ax.set_ylabel('Height (km)')
    ax.set_title('2D Position')
    ax.grid(True)
    ax.legend(handles=ax.get_legend_handles_labels()[0][:2], labels=ax.get_legend_handles_labels()[1][:2])          # This takes only the first two entries of the legend because it repeats
    ax.set_xlim(0, max(odeSolver.x_history[0:last_index_positive]))
    ax.set_ylim(0, odeSolver.h_history[np.argmax(odeSolver.h_history)]*1.1)

    # Save the plot
    plt.savefig('data/output/combined_trajectory.png')

    #------------------------------------------------
    # Trajectory Animation
    #------------------------------------------------
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create quiver plots outside the loop
    flight_arrow = ax.quiver([], [], [], [], scale=25, color='r', label='Body Vector')
    thrust_arrow = ax.quiver([], [], [], [], scale=35, color='b', label='Thrust Vector')

    # Function to update the plot for each frame of animation
    def update(frame):
        i = frame * 500
        x_pos = odeSolver.x_history[i]
        y_pos = odeSolver.h_history[i]
        flight_angle_vector_x = np.cos(odeSolver.theta_history[i] * np.pi / 180)
        flight_angle_vector_y = np.sin(odeSolver.theta_history[i] * np.pi / 180)
        thrust_angle_vector_x = np.cos(odeSolver.psi_history[i] * np.pi / 180)
        thrust_angle_vector_y = np.sin(odeSolver.psi_history[i] * np.pi / 180)
        flight_arrow.set_offsets((x_pos, y_pos))
        flight_arrow.set_UVC(flight_angle_vector_x, flight_angle_vector_y)
        thrust_arrow.set_offsets((x_pos, y_pos))
        thrust_arrow.set_UVC(-thrust_angle_vector_x, -thrust_angle_vector_y)
        return flight_arrow, thrust_arrow

    # Call update_quiver function inside the loop
    ani = FuncAnimation(fig, update, frames=len(odeSolver.t_history) // 500, repeat=True) # interval?
    
    # Set axes labels and title
    ax.set_xlabel('Downrange Position (km)')
    ax.set_ylabel('Height (km)')
    ax.set_title('2D Position')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, max(odeSolver.x_history[0:last_index_positive]))
    ax.set_ylim(0, odeSolver.h_history[np.argmax(odeSolver.h_history)]*1.1)
    ani.save('data/output/combined_trajectory.gif', writer='pillow')

    # Return
    return last_index_positive


if __name__ == '__main__':

    # Init
    values_list = []
    max_down_range_list = []
    max_height_list = []

    # Load in input data
    rocket_params, initial_conditions = load_input_data()

    # PID
    #gain_list = [[26, 91, 80], [51, 88, 99], [0, 0, 1], [0, 0, 0], [75, 1, 0], [97, 0, 3]]
    gain_list = [[51,88,99]]
    for gains in gain_list:

        # Create controller object
        pidController = Controller(gains[0], gains[1], gains[2], rocket_params["DESIRED_FLIGHT_ANGLE"], rocket_params["TVC_BOUNDS"])

        # Generate individual trajectory
        odeSolver = generate_individual_trajectory(pidController, rocket_params, initial_conditions)

        # Create individual plots for each trajectory
        last_index_positive = plot_individual_trajectory(odeSolver)

        # Status
        print("Max Height [km]:",odeSolver.hmax/1e3)
        print(f"Run Complete for: Kp={pidController.Kp}, Ki={pidController.Ki}, Kd={pidController.Kd}")

        values = {"x": odeSolver.x_history, "h": odeSolver.h_history}
        values_list.append(values.copy())
        max_down_range_list.append(values['x'][last_index_positive])
        max_height_list.append(max(values['h'][0:last_index_positive]))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    generation = 1
    for values, gains in zip(values_list, gain_list):
        ax.plot(values['x'], values['h'], label=f"Gen #{generation}: K$_p$={gains[0]}, K$_i$={gains[1]}, K$_d$={gains[2]}")
        generation = generation + 1
    ax.set_xlabel('Downrage (km)')
    ax.set_ylabel('Height (km)')
    ax.set_title('2D Position')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, max(max_down_range_list))
    ax.set_ylim(0, max(max_height_list)*1.1)
    plt.savefig('data/output/dispersed_trajectories.png')




        