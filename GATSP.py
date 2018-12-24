import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import random

matplotlib.rcParams['font.family'] = 'STSong'

# Load the data
city_name = []
city_condition = []
with open('data.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(',')
        city_name.append(line[0])
        city_condition.append([float(line[1]), float(line[2])])
city_condition = np.array(city_condition)

# Distance matrix
city_count = len(city_name)
Distance = np.zeros([city_count, city_count])
for i in range(city_count):
    for j in range(city_count):
        Distance[i][j] = math.sqrt(
            (city_condition[i][0] - city_condition[j][0]) ** 2 + (city_condition[i][1] - city_condition[j][1]) ** 2)

# Population
count = 300
# Number of improvement
improve_count = 10000
# Number of evolution
itter_time = 400

# Set the definition probability of the strong, that is, the first 30% of the population is the strong
retain_rate = 0.3

# Set the survival probability of the weak
random_select_rate = 0.5

# The mutation rate
mutation_rate = 0.1

# Set the starting point
origin = 15
index = [i for i in range(city_count)]
index.remove(15)


# The total distance
def get_total_distance(x):
    distance = 0
    distance += Distance[origin][x[0]]
    for i in range(len(x)):
        if i == len(x) - 1:
            distance += Distance[origin][x[i]]
        else:
            distance += Distance[x[i]][x[i + 1]]
    return distance


# Improved
def improve(x):
    i = 0
    distance = get_total_distance(x)
    while i < improve_count:
        # randint [a,b]
        u = random.randint(0, len(x) - 1)
        v = random.randint(0, len(x) - 1)
        if u != v:
            new_x = x.copy()
            t = new_x[u]
            new_x[u] = new_x[v]
            new_x[v] = t
            new_distance = get_total_distance(new_x)
            if new_distance < distance:
                distance = new_distance
                x = new_x.copy()
        else:
            continue
        i += 1


# Natural selection
def selection(population):
 
    # Sort the total distance from the smallest to the largest
    graded = [[get_total_distance(x), x] for x in population]
    graded = [x[1] for x in sorted(graded)]
    # Pick out the chromosomes that are resilient
    retain_length = int(len(graded) * retain_rate)
    parents = graded[:retain_length]
    # Pick out the chromosomes that are less adaptable, but that survive
    for chromosome in graded[retain_length:]:
        if random.random() < random_select_rate:
            parents.append(chromosome)
    return parents


# Cross breeding
def crossover(parents):
    # The number of progeny generated to ensure population stability
    target_count = count - len(parents)
    # The children list
    children = []
    while len(children) < target_count:
        male_index = random.randint(0, len(parents) - 1)
        female_index = random.randint(0, len(parents) - 1)
        if male_index != female_index:
            male = parents[male_index]
            female = parents[female_index]

            left = random.randint(0, len(male) - 2)
            right = random.randint(left + 1, len(male) - 1)

            # Cross section
            gene1 = male[left:right]
            gene2 = female[left:right]

            child1_c = male[right:] + male[:right]
            child2_c = female[right:] + female[:right]
            child1 = child1_c.copy()
            child2 = child2_c.copy()

            for o in gene2:
                child1_c.remove(o)

            for o in gene1:
                child2_c.remove(o)

            child1[left:right] = gene2
            child2[left:right] = gene1

            child1[right:] = child1_c[0:len(child1) - right]
            child1[:left] = child1_c[len(child1) - right:]

            child2[right:] = child2_c[0:len(child1) - right]
            child2[:left] = child2_c[len(child1) - right:]

            children.append(child1)
            children.append(child2)

    return children


# Mutation
def mutation(children):
    for i in range(len(children)):
        if random.random() < mutation_rate:
            child = children[i]
            u = random.randint(1, len(child) - 4)
            v = random.randint(u + 1, len(child) - 3)
            w = random.randint(v + 1, len(child) - 2)
            child = children[i]
            child = child[0:u] + child[v:w] + child[u:v] + child[w:]


# Get the best pure output
def get_result(population):
    graded = [[get_total_distance(x), x] for x in population]
    graded = sorted(graded)
    return graded[0][0], graded[0][1]


# The population was initialized using an improved loop algorithm
population = []
for i in range(count):
    # Randomly generated individuals
    x = index.copy()
    random.shuffle(x)
    improve(x)
    population.append(x)

register = []
i = 0
distance, result_path = get_result(population)
while i < itter_time:
    # Select breeding groups of individuals
    parents = selection(population)
    # Cross breeding
    children = crossover(parents)
    # Mutation
    mutation(children)
    # Update the population
    population = parents + children

    distance, result_path = get_result(population)
    register.append(distance)
    i = i + 1

result_path = [origin] + result_path + [origin]
print(distance)
print(result_path)


X = []
Y = []
for index in result_path:
    X.append(city_condition[index, 0])
    Y.append(city_condition[index, 1])

plt.plot(X, Y, '-o')
plt.show()

plt.xlabel("Number of iterations")
plt.ylabel("Total moving distance")
plt.plot(list(range(len(register))), register)
plt.show()
