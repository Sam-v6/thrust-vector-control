#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 5)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    # Apply randomness
    random.seed(64)

    # Starting population size
    pop = toolbox.population(n=30)

    # Setting algo and stats
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    stats.register("fitness", lambda pop: [ind for ind in pop])
    stats.register("genes", lambda genes: [ind for ind in pop])
    
    # Calling the algo
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, 
                                   stats=stats, halloffame=hof, verbose=False)

    # Print the best one
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    # Return
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()

    # Get generation and participants
    generation_count = len(log)-1
    print("Total Generations", generation_count)

    for generation in range(0,len(log)):
        print("Generation: ", generation+1)
        for individual in range(0,len(log[generation]["genes"])):
            print("Individual:", individual+1, "Fitness:", log[generation]["fitness"][individual][0], "Values:", log[generation]["genes"][individual][0],log[generation]["genes"][individual][1],log[generation]["genes"][individual][2])


    # print(log[0]["fitness"][0][0])  # Generation, Key, Participant, Integer to get it out of a tuple
    # print(log[0]["genes"][0])
    # print(log[0]["genes"][0][0], log[0]["genes"][0][1], log[0]["genes"][0][2])      # Geeneration, Key, Participant, Value