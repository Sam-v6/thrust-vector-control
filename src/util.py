# Pkg imports
import math
import yaml
import numpy as np

def load_input_data():

    #-------------------------------------------------------------------------------------------------
    # Rocket parameter data
    #-------------------------------------------------------------------------------------------------
    with open('data/input/rocket_parameters.yaml', 'r') as file:
        # Load the contents of the file into a dictionary
        rocket_params = yaml.safe_load(file)

    # Update some rocket parameters (derived input data)
    rocket_params["PROFILE_AREA"] = np.pi/(4*rocket_params["DIAMETER"]**2)
    rocket_params["MASS_PROP_DESCENT"] = rocket_params["MASS_STRUCTURE"] * rocket_params["MASS_PROP_DESCENT_RATIO"]
    rocket_params["MASS_INIT"] = rocket_params["MASS_PROP_ASCENT"] + rocket_params["MASS_STRUCTURE"] + rocket_params["MASS_PAYLOAD"] + rocket_params["MASS_PROP_DESCENT"]
    rocket_params["MDOT_ASCENT"] =  rocket_params["MASS_PROP_ASCENT"]/rocket_params["TIME_ASCENT_BURN"]
    rocket_params["MDOT_DESCENT"] =  rocket_params["MDOT_ASCENT"]/rocket_params["DESCENT_RATIO"]
    rocket_params["THRUST_DESCENT"] = rocket_params["THRUST_ASCENT"]/rocket_params["DESCENT_RATIO"]

    #-------------------------------------------------------------------------------------------------
    # Initial Conditions
    #-------------------------------------------------------------------------------------------------
    with open('data/input/initial_conditions.yaml', 'r') as file:
        # Load the contents of the file into a dictionary
        initial_conditions = yaml.safe_load(file)
    
    # Convert to radians
    initial_conditions["theta_init"] = np.radians(initial_conditions["theta_init"])

    # Return
    return rocket_params, initial_conditions 

def normalize_radians(angle):
    while angle < -math.pi:
        angle += 2 * math.pi
    while angle >= math.pi:
        angle -= 2 * math.pi
    return angle