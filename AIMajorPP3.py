# Genetic Algorithm for Travelling Salesman Problem
# By Lauren Dennedy and Matthew Stanford
# gr5743 and gd9687

# General Overview:

# This genetic algorithm will run for a specified amount of generations,
# with a set population size for each generation.
# The starting solutions for the first generation are all randomly
# generated. Each generation after that uses the offspring of the previous
# generation to crossover and mutate to create new solutions for the next one.

# The default values for this algorithm are:
# Starts at city 1
# Each population has 20 individuals
# 10 generations are created
# There is a 10% chance for mutation

# Change the values at the bottom of the code to see
# different results. You can also change which city you start at
# by changing the value for the starting state at the bottom
# from the list of states available.

# State Representation/Encoding Scheme:

# States are a list of mutable coordinate pairs, with a boolean for
# if it has been visited, then another for if it is the current city we're at.
# There are 2 elements, one at the very beginning, and one at the very end, to denote
# which city we're at, and the total distance travelled.

# Each list is a starting state for if we started at each city.
# Right now, only start_1 is being used, as the default city. You can change which
# state to start at at the bottom by changing the number for the city to start at
# in the initial population creation.
start_1 = [1, [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], 0]
start_2 = [2, [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], 0]
start_3 = [3, [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], 0]
start_4 = [4, [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], 0]
start_5 = [5, [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], 0]
start_6 = [6, [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], 0]
start_7 = [7, [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], 0]
start_8 = [8, [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], 0]
start_9 = [9, [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], 0]

# The goal state needs to be broad enough to match any starting city, so once all cities
# have been visited, the start and distance is set back to 0 when done.
goal = [0, [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], 0]

# Issues:

# Some logical inconsistencies will appear at higher
# generation and population values, such as "backwards" distances.
# This may be due to some of the swapping features in the crossover and
# mutation functions having untested edge cases that
# we didn't have time to test.

# The solution does not return to the starting city as expected. It should visit
# each city once and only once, but returning to the start is not implemented.
# We wrote a majority of our code before knowing it was required, then ran out of
# time to add this final feature, which would require reworking many of our
# functions.

# This was imported to solve the issue of the population states
# being the same everytime, so, a *deep copy* is necessary
from copy import deepcopy

from collections import Counter

import random

# A dictionary to store values of distances between 2 cities
distances = {
    (1, 2): 2,
    (1, 3): 11,
    (1, 4): 3,
    (1, 5): 18,
    (1, 6): 14,
    (1, 7): 20,
    (1, 8): 12,
    (1, 9): 5,
    (2, 3): 13,
    (2, 4): 10,
    (2, 5): 5,
    (2, 6): 3,
    (2, 7): 8,
    (2, 8): 20,
    (2, 9): 17,
    (3, 4): 5,
    (3, 5): 19,
    (3, 6): 21,
    (3, 7): 2,
    (3, 8): 5,
    (3, 9): 8,
    (4, 5): 6,
    (4, 6): 4,
    (4, 7): 12,
    (4, 8): 15,
    (4, 9): 1,
    (5, 6): 12,
    (5, 7): 6,
    (5, 8): 9,
    (5, 9): 7,
    (6, 7): 19,
    (6, 8): 7,
    (6, 9): 4,
    (7, 8): 21,
    (7, 9): 13,
    (8, 9): 6
}


# Define creating the population
# How can we create the population? The initial population is going to be a bunch
# of random solutions, meaning they all solve the problem with multiple states
# between steps, just randomly generated

# Each individual within the population needs to be a list of states, because each
# individual is a possible solution
def population(current_state, goal_state, size=100):
    # current_state: where the solution should start from
    # goal_state: where the solution should end
    # size: how many solutions are in the population (default 100)

    # Setup to generate a random solution
    goal = False
    check = True
    population = []

    # Deepcopy is necessary to ensure new solutions are generated,
    # instead of all overwriting the previous one
    starting_state = deepcopy(current_state)

    # Create size amount of solutions
    for i in range(size):
        solution = []
        solution.append(starting_state)
        current_state = deepcopy(starting_state)

        next_state = []
        goal = False
        check = True

        # Loop to create "actions" that get appended to solution until the final state = goal
        while not goal:

            check = True
            visited_cities = []
            current_city = -1
            for j in range(1, 10):
                if current_state[j][0] == 1:
                    visited_cities.append(j)
                if current_state[j][1] == 1:
                    current_city = j

            next_city = random.randint(1, 9)
            while next_city == current_city or next_city in visited_cities:
                next_city = random.randint(1, 9)

            # Create a new state based on the next city
            next_state = deepcopy(current_state)
            for k in range(1, 10):
                # The old "current" city
                if next_state[k][1] == 1:
                    # Change it to 0, we're not there anymore
                    next_state[k][1] = 0
                # Mark the new "current city", where we're moving to
                if k == next_city:
                    next_state[k][1] = 1
                    next_state[k][0] = 1
                for c in visited_cities:
                    next_state[c][0] = 1

            # Lastly, change the distance, based on the current distance, + some from change
            current_distance = current_state[10]
            # From the current city to the next city
            # Get this data from the dictionary
            # If the order is wrong/backwards, flip it to get the distance in the other direction
            try:
                added_distance = distances[(current_city, next_city)]
            except KeyError:
                added_distance = distances[(next_city, current_city)]
            new_distance = current_distance + added_distance
            next_state[10] = new_distance

            current_state = next_state

            # Need to check if we're nearly at the goal state
            for l in range(1, 10):
                # If there is any city we haven't been to yet,
                if current_state[l][0] == 0:
                    # We're not at our goal
                    check = False

            if check == True:
                # If check remains true, then we're at the goal
                goal = True

            # Make it match the goal if we reach the end
            if goal:
                last_state = deepcopy(current_state)
                solution.append(current_state)
                last_state[0] = 0
                last_state[10] = 0
                last_state[next_city][1] = 0
                solution.append(last_state)
            else:
                # Not at the goal yet, append the state as is
                solution.append(current_state)

        population.append(solution)

    return population


# Builds population from the offspring
# The offspring use the crossover and mutation functions together to build the next
# generation
def build_refined_population(offspring, size=100):
    new_population = deepcopy(offspring)
    new_offspring = []
    new_parents = []

    # Randomly pick new parents and keep crossing over to generate new individuals for
    # the next population

    for s in range(int(size / 2)):
        new_offspring = []
        new_parents = []
        parent_1_index = random.randint(0, len(new_population) - 1)
        parent_2_index = random.randint(0, len(new_population) - 1)
        while parent_1_index == parent_2_index:
            parent_2_index = random.randint(0, len(new_population) - 1)

        parent_1 = new_population[parent_1_index]
        parent_2 = new_population[parent_2_index]
        new_parents.append(parent_1)
        new_parents.append(parent_2)

        new_offspring = crossover(new_parents)
        for child in new_offspring:
            new_population.append(child)

    return new_population


def fitness(adult):
    # Fitness is determined by how short the total distance is
    # The total distance is the last element in the second to last row
    return adult[-2][10]


def repair_solution(solution):
    # Take a solution that might be broken
    # Go through the changes between steps, see if they make sense
    # Change if they do not.

    # When its broken, we know that there could be a missing city

    # First, find order of all cities
    all_cities = []
    for i in range(0, 10):
        for j in range(1, 10):
            if solution[i][j] == [1, 1]:
                all_cities.append(j)

    # Find which cities are unique from those
    unique_cities = Counter(all_cities).keys()
    missing_cities = []
    duplicate_cities = []
    city_range = range(1, 10)

    # Missing cities will not be in the unique cities set
    for city in city_range:
        if city not in unique_cities:
            missing_cities.append(city)

    # Find duplicate cities
    for city in all_cities:
        if all_cities.count(city) > 1:
            duplicate_cities.append(city)

    # Only unique duplicates
    duplicate_cities = Counter(duplicate_cities).keys()

    # How to rebuild states with this knowledge?
    # Everytime we run into a duplicate, swap it with the next missing one
    # The number of missing cities will equal the number of duplicate cities

    duplicate_indices = []
    for dupe in duplicate_cities:
        duplicate_indices.append(all_cities.index(dupe))

    swap_count = 0
    for dupe_index in duplicate_indices:
        solution[dupe_index][list(duplicate_cities)[swap_count]][0] = 0
        solution[dupe_index][list(duplicate_cities)[swap_count]][1] = 0
        solution[dupe_index][missing_cities[swap_count]][0] = 1
        solution[dupe_index][missing_cities[swap_count]][1] = 1
        swap_count += 1

    # After this point, city orders are repaired, but the memory of cities visited
    # may be incorrect
    # We have to repair the memory of where cities have been
    # Because building new populations depends on memory of order

    # Get new all city information
    all_cities = []
    for i in range(0, 10):
        for j in range(1, 10):
            if solution[i][j] == [1, 1]:
                all_cities.append(j)

    # Rebuilding the memory
    # Go through all the states
    # Set previous visitation to 0 - clear memory of where we've been
    for state in solution:
        for i in range(1, 10):
            state[i][0] = 0

    # We know the order of the cities
    # Then bring down the memory of knowing where we have travelled through each state
    cities = 0
    cities_to_change = []
    cities_to_change.append(all_cities[0])
    for state in solution:
        for city in all_cities:
            for c in cities_to_change:
                state[c][0] = 1
        cities += 1
        if cities > 8:
            cities = 8
        cities_to_change.append(all_cities[cities])

    # We can use all_cities to calculate the new distance

    solution_distances = []

    current_distance = solution[0][10]
    solution_distances.append(current_distance)

    for x in range(0, 8):
        try:
            distance = distances[(all_cities[x], all_cities[x + 1])]
        except KeyError:
            try:
                distance = distances[(all_cities[x + 1], all_cities[x])]
            except KeyError:
                distance = current_distance
        current_distance += distance
        solution_distances.append(current_distance)

    dist_count = 0
    for state in solution:
        state[10] = solution_distances[dist_count]
        dist_count += 1
        if dist_count > 8:
            dist_count = 8
            solution[9][10] = 0

    # After this, city information should be repaired by making sure:
    # 1. There are no missing cities or duplicates
    # 2. The travel memory has been rebuilt and restored
    # 3. The distance has been recalculated to fit the new city order


# Crossover genes of parents to create 2 children
# 2 children are necessary since they will be the first 2 parents of the next generation
def crossover(parents):
    children = []
    child_1 = []
    child_2 = []

    # Set a random split point between the two parents
    # at least 1 city from each parent will be used
    split_point = random.randint(2, 8)

    child_1 = deepcopy(parents[0])
    child_2 = deepcopy(parents[1])
    for i in range(1, split_point):
        child_1[i] = parents[1][i]
        child_2[10 - i] = parents[0][10 - i]

    # Split the cities
    # ignore distance while splitting cities / leave it incorrect,
    # because we can repair any inconsistencies with our repair function
    repair_solution(child_1)
    repair_solution(child_2)

    child_1 = mutate(child_1)
    child_2 = mutate(child_2)

    children.append(child_1)
    children.append(child_2)

    return children


def mutate(child):
    # Create a random chance to mutate each child
    # We have to specify what that chance is and how it is being modified
    # What will we change?
    # Just mutates 1 gene in the child
    # Change one of the cities to another city

    # First get the full city order from the child
    all_cities = []
    for i in range(0, 10):
        for j in range(1, 10):
            if child[i][j] == [1, 1]:
                all_cities.append(j)

    # 10% chance to mutate
    chance = random.uniform(0, 1)
    mutate = False
    if chance <= 0.1:
        mutate = True

    # If 0.1 or less, mutation will succeed
    if mutate:
        swap_1 = random.randint(2, 8)
        swap_2 = random.randint(2, 8)
        while swap_2 == swap_1:
            swap_2 = random.randint(2, 8)

        swap_1_index = all_cities.index(swap_1)
        swap_2_index = all_cities.index(swap_2)

        temp_state = child[swap_1_index]
        child[swap_1_index] = child[swap_2_index]
        child[swap_2_index] = temp_state

        repair_solution(child)

    return child


# The genetic algorithm
# Uses all of the functions together to generate solutions
def genetic_algorithm(population, population_size=100, rounds=100):
    # population: The initial population of solutions, all randomly generated
    # population_size: The size of each generation of solutions (default 100)
    # rounds: How many generations to iterate through (default 100)

    # Choose 2 adults with highest fitness values
    # If more than 2 with same fitness value, randomly select 2 from those
    # Children is part of the new population
    # Maybe we can call population again based on the children
    # Call population, create a new population based on the children, then repeat, until either the individual is fit enough, or enough time has elapsed

    # For each generation:
    for r in range(0, rounds):
        print("Generation", r)

        # First, we want to know the two best adults to reproduce
        best_solutions = []
        minimum = 200
        minimum_2 = 200
        for solution in population:
            fit_num = fitness(solution)
            if fit_num < minimum:
                minimum = fit_num

        for solution in population:
            if solution[-2][10] == minimum:
                best_solutions.append(solution)

        for solution in population:
            fit_num = fitness(solution)
            if fit_num < minimum_2:
                if fit_num != minimum:
                    minimum_2 = fit_num

        for solution in population:
            if solution[-2][10] == minimum_2:
                best_solutions.append(solution)

        # After this is done we should have the two best solutions

        # Create the children using crossover and mutate functions
        solution_children = crossover(best_solutions)

        new_population = []
        print("Creating new generation")
        new_population = build_refined_population(solution_children, population_size)
        population = deepcopy(new_population)

    # After all of the generations have been completed, find the best solution from the last generation
    best_solution = []
    minimum = 200
    for solution in new_population:
        fit_num = fitness(solution)
        if fit_num < minimum:
            minimum = fit_num

    for solution in new_population:
        if solution[-2][10] == minimum:
            best_solution = solution

    print(rounds, " generations have elapsed.")
    print("The best solution found was the following:")
    for state in best_solution:
        print(state)

    print("City travel order is as follows:")
    all_cities = []
    for i in range(0, 10):
        for j in range(1, 10):
            if best_solution[i][j] == [1, 1]:
                all_cities.append(j)

    print(all_cities)

    print("Total distance travelled:", best_solution[-2][10])

    # Then return the best solution
    return best_solution


# You can change which city to start at by changing the value here in the population creation
salesman_pop = population(start_1, goal, 20)

# Run the genetic algorithm with 20 individuals each generation, for 10 generations
genetic_algorithm(salesman_pop, 20, 10)