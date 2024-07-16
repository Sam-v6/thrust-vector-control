"""
Purpose: Main driver for thrust vector control of rocket's descent stage
Author: Syam Evani, Summer 2024
"""

# Standard library imports
import sys
import shutil
import os
import random
import array
import multiprocessing
import time
from functools import partial

# Additional library imports
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as mpatches
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# Local imports
from controller import Controller
from ode_solver import OdeSolver
from util import load_input_data, convert_to_normalized_degrees, flip_angle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.plotter.plotter import Plotter

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

def post_process_individual_trajectory(odeSolver, bounds):

    #------------------------------------------------
    # Post Processing
    #------------------------------------------------

    # Convert to km/s, km, and Metric Tons, and kN
    for i in range(0,len(odeSolver.t_history)):
        odeSolver.x_history[i] = odeSolver.x_history[i]/1e3                       # [km]
        odeSolver.h_history[i] = odeSolver.h_history[i]/1e3                       # [km]
        odeSolver.v_history[i] = odeSolver.v_history[i]/1e3                       # [km/s]
        odeSolver.F_history[i] = odeSolver.F_history[i]/1e3                       # [kN]
        odeSolver.m_history[i] = odeSolver.m_history[i]/1e3                       # [MT]
        odeSolver.mp_descent_history[i] = odeSolver.mp_descent_history[i]/1e3     # [MT]
        odeSolver.mp_ascent_history[i] = odeSolver.mp_ascent_history[i]/1e3       # [MT]

        # Create the controller bounds
        odeSolver.controller_upper_bound.append(odeSolver.theta_history[i] + bounds)
        odeSolver.controller_lower_bound.append(odeSolver.theta_history[i] - bounds)

    # Post processing the height array
    h_array = np.array(odeSolver.h_history)
    reversed_h_array = h_array[::-1]
    last_index_positive = len(h_array) - np.argmax(reversed_h_array > 0) - 1

    return odeSolver, last_index_positive

def plot_individual_trajectory(odeSolver, last_index_positive):

    #------------------------------------------------
    # Create directories
    #------------------------------------------------
    new_folder = f"{odeSolver.controller.Kp}_{odeSolver.controller.Ki}_{odeSolver.controller.Kd}"
    new_dir_path = os.path.join(os.environ.get('USERPROFILE'), 'repos', 'thrust-vector-control', 'data', 'output', new_folder)
    os.makedirs(new_dir_path, exist_ok=True)

    # Close all plots
    plt.close('all')

    #------------------------------------------------
    # Plotter implementation
    #------------------------------------------------
    # Single plot configuration with xlim and ylim
    phase_plot = {
        'datasets': [{'x_data': odeSolver.t_history, 'y_data': odeSolver.phase_history, 'label': "Phase"}],
        'title': "Phase",
        'xlabel': "Time (s)",
        'ylabel': "Phase",
        'xlim': [0, odeSolver.t_history[last_index_positive]], 
        'plot': [0, 0]
    }

    height_plot = {
        'datasets': [{'x_data': odeSolver.t_history, 'y_data': odeSolver.h_history, 'label': "Height"}],
        'title': "Height",
        'xlabel': "Time (s)",
        'ylabel': "Height (km)",
        'xlim': [0, odeSolver.t_history[last_index_positive]], 
        'ylim': [0, max(odeSolver.h_history[0:last_index_positive])*1.1],
        'plot': [1, 0]  
    }

    downrange_plot = {
        'datasets': [{'x_data': odeSolver.t_history, 'y_data': odeSolver.x_history, 'label': "Downrange"}],
        'title': "Downrange",
        'xlabel': "Time (s)",
        'ylabel': "Downrange (km)",
        'xlim': [0, odeSolver.t_history[last_index_positive]], 
        'ylim': [0, max(odeSolver.x_history[0:last_index_positive])*1.1],
        'plot': [2, 0]  
    }

    velocity_plot = {
        'datasets': [{'x_data': odeSolver.t_history, 'y_data': odeSolver.v_history, 'label': "Velocity"}],
        'title': "Velocity",
        'xlabel': "Time (s)",
        'ylabel': "Velocity (km/s)",
        'xlim': [0, odeSolver.t_history[last_index_positive]], 
        'ylim': [0, max(odeSolver.v_history[0:last_index_positive])*1.1],
        'plot': [3, 0]  
    }

    flight_angles_plot = {
        'datasets': [{'x_data': odeSolver.t_history, 'y_data': odeSolver.theta_history, 'label': "$\\theta$"},
                     {'x_data': odeSolver.t_history, 'y_data': odeSolver.theta_error_history, 'label': "$\\theta_e$"},
                     {'x_data': odeSolver.t_history, 'y_data': odeSolver.psi_history, 'label': "$\\psi$"},
                     {'x_data': odeSolver.t_history, 'y_data': odeSolver.controller_upper_bound, 'label': "Controller Upper Bound", 'color': "red", 'linestyle': "--"},
                     {'x_data': odeSolver.t_history, 'y_data': odeSolver.controller_lower_bound, 'label': "Controller Lower Bound", 'color': "lightcoral", 'linestyle': "--"}],
        'title': "Flight Angles",
        'xlabel': "Time (s)",
        'ylabel': "Angle (deg)",
        'xlim': [0, odeSolver.t_history[last_index_positive]], 
        'plot': [4, 0]  
    }

    mass_plot = {
        'datasets': [{'x_data': odeSolver.t_history, 'y_data': odeSolver.m_history, 'label': "Total Mass"},
                     {'x_data': odeSolver.t_history, 'y_data': odeSolver.mp_descent_history, 'label': "Descent Propellant Mass"},
                     {'x_data': odeSolver.t_history, 'y_data': odeSolver.mp_ascent_history, 'label': "Ascent Propellant Mass"}],
        'title': "Mass",
        'xlabel': "Time (s)",
        'ylabel': "Mass (MT)",
        'xlim': [0, odeSolver.t_history[last_index_positive]], 
        'plot': [5, 0]  
    }

    thrust_plot = {
        'datasets': [{'x_data': odeSolver.t_history, 'y_data': odeSolver.F_history, 'label': "Thrust"}],
        'title': "Thrust",
        'xlabel': "Time (s)",
        'ylabel': "Thrust (kN)",
        'xlim': [0, odeSolver.t_history[last_index_positive]], 
        'plot': [6, 0]  
    }

    density_plot = {
        'datasets': [{'x_data': odeSolver.t_history, 'y_data': odeSolver.rho_history, 'label': "Density"}],
        'title': "Density",
        'xlabel': "Time (s)",
        'ylabel': "Density (kg/m$^3$)",
        'xlim': [0, odeSolver.t_history[last_index_positive]], 
        'plot': [7, 0]  
    }

    gravity_plot = {
        'datasets': [{'x_data': odeSolver.t_history, 'y_data': odeSolver.g_history, 'label': "Gravity"}],
        'title': "Gravity",
        'xlabel': "Time (s)",
        'ylabel': "Gravity (m/s$^2$)",
        'xlim': [0, odeSolver.t_history[last_index_positive]], 
        'plot': [8, 0]  
    }

    # Create a Plotter instance with a single plot configuration
    combinedPlots = Plotter(os.path.join(new_dir_path,"plotter_combined_plots.png"), 
                                    phase_plot,
                                    height_plot, 
                                    downrange_plot, 
                                    velocity_plot,
                                    flight_angles_plot,
                                    mass_plot,
                                    thrust_plot,
                                    density_plot,
                                    gravity_plot)
    combinedPlots.plot()

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
    axs[1,1].plot(odeSolver.t_history, odeSolver.controller_upper_bound, label="Controller Upper Bound", color="red", linestyle="--")
    axs[1,1].plot(odeSolver.t_history, odeSolver.controller_lower_bound, label="Controller Lower Bound", color="lightcoral", linestyle="--")
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
    plt.savefig(os.path.join(new_dir_path,'combined_plots.png'))
    plt.cla()
    plt.close(fig)

    #------------------------------------------------
    # Trajectory Plot
    #------------------------------------------------
    # Find the indexes where the phase changes
    phase_history = odeSolver.phase_history
    change_indexes = [i for i in range(1, len(phase_history)) if phase_history[i] != phase_history[i-1]]

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(odeSolver.x_history, odeSolver.h_history, color='gray', linestyle="--")

    # Add vertical lines at phase change points
    for index in change_indexes:
        ax.axvline(x=odeSolver.x_history[index], color='black', linestyle=':')

    for i in range(0, len(odeSolver.t_history), 500):
            
            # Flight
            flight_angle_vector_x = np.cos(odeSolver.theta_history[i] * np.pi / 180)
            flight_angle_vector_y = np.sin(odeSolver.theta_history[i] * np.pi / 180)
            ax.quiver(odeSolver.x_history[i], odeSolver.h_history[i], flight_angle_vector_x, flight_angle_vector_y, scale=35, color='b', label='Flight Direction')    

            # Plume
            plume_angle_vector_x = -np.cos(odeSolver.psi_history[i] * np.pi / 180)
            plume_angle_vector_y = -np.sin(odeSolver.psi_history[i] * np.pi / 180)
            ax.quiver(odeSolver.x_history[i], odeSolver.h_history[i], plume_angle_vector_x, plume_angle_vector_y, scale=35, color='r', label='Plume')
        
    ax.set_xlabel('Downrange Position (km)')
    ax.set_ylabel('Height (km)')
    ax.set_title('2D Position')
    ax.legend(handles=ax.get_legend_handles_labels()[0][:2], labels=ax.get_legend_handles_labels()[1][:2])          # This takes only the first two entries of the legend because it repeats
    
    # Add major and minor grid lines
    ax.grid(True, which='both')
    ax.grid(which='major', linestyle='-', linewidth='0.4', color='gray')
    ax.grid(which='minor', linestyle=':', linewidth='0.4', color='gray')
    
    # Increase minor ticks for a cleaner grid appearance
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(0, max(odeSolver.x_history[0:last_index_positive])*1.1)
    ax.set_ylim(0, odeSolver.h_history[np.argmax(odeSolver.h_history)]*1.1)

    # Save the plot
    plt.savefig(os.path.join(new_dir_path,'combined_trajectory.png'))
    plt.close(fig)

    #------------------------------------------------
    # Trajectory Animation
    #------------------------------------------------
    # Create phase map
    phase_map = {1: "Ascent",
                 2: "Ascent with Gravity Turn",
                 3: "Coast",
                 4: "Descent",
                 5: "Free Fall"}

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create quiver plots outside the loop
    plume_arrow = ax.quiver([], [], [], [], scale=35, color='b', label='Plume Vector')

    # Create a text object for displaying phase with a background color
    phase_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top',
                        bbox=dict(facecolor='lightblue', alpha=1.0, edgecolor='black'))

    # Function to update the plot for each frame of animation
    scale_multiplier = 250
    def update(frame):
        i = frame * scale_multiplier
        x_pos = odeSolver.x_history[i]
        y_pos = odeSolver.h_history[i]

        # Plume
        plume_angle_vector_x = -np.cos(odeSolver.psi_history[i] * np.pi / 180)
        plume_angle_vector_y = -np.sin(odeSolver.psi_history[i] * np.pi / 180)
        plume_arrow.set_offsets((x_pos, y_pos))

        plume_arrow.set_UVC(plume_angle_vector_x, plume_angle_vector_y)

        # Update the phase text
        phase = phase_map[odeSolver.phase_history[i]]
        phase_text.set_text(f'Phase: {phase}')

        # Return
        return plume_arrow, phase_text

    # Call update_quiver function inside the loop
    ani = FuncAnimation(fig, update, frames=len(odeSolver.t_history) // scale_multiplier, interval=100, repeat=True) # interval?
    
    # Set axes labels and title
    ax.set_xlabel('Downrange Position (km)')
    ax.set_ylabel('Height (km)')
    ax.set_title('2D Position')
    ax.legend()

    # Add major and minor grid lines
    ax.grid(True, which='both')
    ax.grid(which='major', linestyle='-', linewidth='0.4', color='gray')
    ax.grid(which='minor', linestyle=':', linewidth='0.4', color='gray')
    
    # Increase minor ticks for a cleaner grid appearance
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(0, max(odeSolver.x_history[0:last_index_positive])*1.1)
    ax.set_ylim(0, odeSolver.h_history[np.argmax(odeSolver.h_history)]*1.1)

    # Save
    ani.save(os.path.join(new_dir_path,'combined_trajectory.gif'), writer='pillow')

    # Return
    return 0

def process_gains(gains, rocket_params, initial_conditions):

    # Create controller object
    pidController = Controller(gains[0], gains[1], gains[2], rocket_params["DESIRED_FLIGHT_ANGLE"], rocket_params["TVC_BOUNDS"])

    # Generate individual trajectory
    odeSolver = generate_individual_trajectory(pidController, rocket_params, initial_conditions)

    # Create individual plots for each trajectory
    odeSolver, last_index_positive = post_process_individual_trajectory(odeSolver, rocket_params["TVC_BOUNDS"])
    plot_individual_trajectory(odeSolver, last_index_positive)

    # Status
    print(f"Run Complete for: Kp={pidController.Kp}, Ki={pidController.Ki}, Kd={pidController.Kd}")

    # Store values into a dictionary
    values = {"x": odeSolver.x_history, "h": odeSolver.h_history}

    # Create lists
    max_down_range = values['x'][last_index_positive]
    max_height = max(values['h'][0:last_index_positive])

    return values, max_down_range, max_height

def biased_attr_kp():
    return random.randint(0, 100) 

def biased_attr_ki():
    # Randomly choose whether Ki should be 0 or non-zero
    if random.random() < 0.3:
        return 0
    else:
        return random.randint(1, 100)  # Non-zero value within bounds

def biased_attr_kd():
    return random.randint(0, 100)  

def evalOneMax(individual, rocket_params, initial_conditions):
    # Create controller object
    pidController = Controller(individual[0], individual[1], individual[2], rocket_params["DESIRED_FLIGHT_ANGLE"], rocket_params["TVC_BOUNDS"])

    # Generate individual trajectory
    odeSolver = generate_individual_trajectory(pidController, rocket_params, initial_conditions)

    # Create individual plots for each trajectory
    odeSolver, last_index_positive = post_process_individual_trajectory(odeSolver)
    plot_individual_trajectory(odeSolver, last_index_positive)

    # Return
    return (odeSolver.x_history[last_index_positive],)  # Must intentionally return as a tuple
class Evaluator:
    def __init__(self, rocket_params, initial_conditions):
        self.rocket_params = rocket_params
        self.initial_conditions = initial_conditions

    def __call__(self, individual):
        return evalOneMax(individual, self.rocket_params, self.initial_conditions)

def run_genetic_algo(rocket_params, initial_conditions, CORE_COUNT):
    # Multiprocessing pool
    pool = multiprocessing.Pool(CORE_COUNT)

    # Create objects
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

    # Toolboxes
    toolbox = base.Toolbox()
    toolbox.register("attr_kp", biased_attr_kp)
    toolbox.register("attr_ki", biased_attr_ki)
    toolbox.register("attr_kd", biased_attr_kd)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_kp, toolbox.attr_ki, toolbox.attr_kd), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Initialize evaluator
    evaluator = Evaluator(rocket_params, initial_conditions)
    toolbox.register("evaluate", evaluator)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Use the pool's map method
    toolbox.register("map", pool.map)

    # Setup
    random.seed(64)
    pop = toolbox.population(n=10)  # Initial pop size

    # Setting algo and stats
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("fitness", lambda pop: [ind for ind in pop])
    stats.register("genes", lambda genes: [ind for ind in pop])

    # Calling the algo
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.6, ngen=5,  # Generations
                                   stats=stats, halloffame=hof, verbose=True)

    # Print the best one
    best_ind = tools.selBest(pop, 1)[0]

    # Close the pool
    pool.close()
    pool.join()

    # Return
    return pop, log, hof, best_ind

if __name__ == '__main__':

    #--------------------------------------------------
    # Start timer and set inputs
    #--------------------------------------------------
    # Start timing the original version
    start_time = time.time()

    # Input
    ENABLE_GENETIC_ALGO = False
    CORE_COUNT = int(multiprocessing.cpu_count()/2)

    #--------------------------------------------------
    # Perform housekeeping and load in data
    #--------------------------------------------------
    # Clean up everything below /data/output/
    output_path = os.path.join(os.environ.get('USERPROFILE'), 'repos', 'thrust-vector-control', 'data', 'output')
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

        # Recreate the directory
        os.makedirs(output_path)

    # Load in input data
    rocket_params, initial_conditions = load_input_data()

    #--------------------------------------------------
    # Run genetic algo if enabled
    #--------------------------------------------------
    if ENABLE_GENETIC_ALGO:

        # Calling genetic algo
        pop, log, hof, best_ind = run_genetic_algo(rocket_params, initial_conditions, CORE_COUNT)
        
        # Print information about GA
        for generation in range(0,len(log)):
            print("Generation: ", generation+1)
            for individual in range(0,len(log[generation]["genes"])):
                print("Individual:", individual+1, "Fitness:", log[generation]["fitness"][individual][0], "Values:", log[generation]["genes"][individual][0],log[generation]["genes"][individual][1],log[generation]["genes"][individual][2])

        # Print best individual
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

        # Function to update the plot
        def update_plot(frame):
            ax.cla()  # Clear the previous plot
            ax.set_xlabel('K$_p$')
            ax.set_ylabel('K$_i$')
            ax.set_zlabel('K$_d$')
            ax.set_title(f'Generational Gain Selections (Generation {frame+1})')
            ax.set_xlim([0, 100])  # Adjust limits as needed
            ax.set_ylim([0, 100])
            ax.set_zlim([0, 100])
            
            colors = ['blue', 'red', 'green', 'orange', 'magenta', 'lime' ]  # Define colors for each generation
            generation_labels = []
            for individual in range(len(log[frame]["genes"])):
                x = log[frame]["genes"][individual][0]
                y = log[frame]["genes"][individual][1]
                z = log[frame]["genes"][individual][2]
                ax.scatter(x, y, z, c=colors[frame], marker='o', s=40)

            # Add legend
            legend_handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors[:frame+1], generation_labels)]
            ax.legend(handles=legend_handles, loc='upper left')

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Generate the animation
        ani = animation.FuncAnimation(fig, update_plot, frames=len(log), interval=1000)

        # Save the animation as a GIF
        ani.save(os.path.join(os.environ.get('USERPROFILE'), 'repos', 'thrust-vector-control', 'data', 'output', 'generational_gains_animation.gif'), writer='pillow', fps=1)

    #--------------------------------------------------
    # Run manual gains
    #--------------------------------------------------
    else:
        # Init
        values_list = []
        max_down_range_list = []
        max_height_list = []

        # PID
        #gain_list = [[26, 91, 80], [51, 88, 99], [0, 0, 1], [0, 0, 0], [75, 1, 0], [97, 0, 3]]
        gain_list = [[22, 0, 0]]
        
        # Create a Pool of workers
        with multiprocessing.Pool(processes=CORE_COUNT) as pool:
            # Map the process_gains function to the list of gains
            results = pool.starmap(process_gains, [(gains, rocket_params, initial_conditions) for gains in gain_list])

        # Unpack results
        for values, max_down_range, max_height in results:
            values_list.append(values)
            max_down_range_list.append(max_down_range)
            max_height_list.append(max_height)

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
        plt.savefig(os.path.join(os.environ.get('USERPROFILE'), 'repos', 'thrust-vector-control', 'data', 'output', 'dispersed_trajectories.png'))
        plt.close(fig)

    # End timing the original version
    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration: {duration:.2f} seconds")






        