# Pkg imports
import math

def normalize_radians(theta):
    while theta < 0:
        theta += 2 * math.pi
    while theta >= 2 * math.pi:
        theta -= 2 * math.pi
    return theta