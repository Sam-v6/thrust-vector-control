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

# def normalize_radians(angle):
#     while angle < -math.pi:
#         angle += 2 * math.pi
#     while angle >= math.pi:
#         angle -= 2 * math.pi
#     return angle

def normalize_radians(radians):
    return radians % (2 * math.pi)

def convert_to_normalized_radians(angle):
    radians = np.radians(angle)
    return normalize_radians(radians)

def convert_to_normalized_degrees(radians):
    # Convert radians to degrees
    degrees = np.degrees(radians) % 360
    # Normalize to the range -180 to 180
    #signed_degrees = ((degrees + 180) % 360) - 180
    return degrees

def flip_angle(degrees):
    degrees = degrees % 360     # make sure its normalized
    delta = 180 - abs(degrees)

    # Flip angle if positive (if negative, it will be positive already from above)
    if degrees > 0:
        delta = delta * -1

    # Return
    return delta      