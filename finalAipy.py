import numpy as np
import random
import matplotlib.pyplot as plt

import numpy as np 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
#import statsmodels.api as sm

import pandas as pd


import seaborn as sns
#%matplotlib inline


data = pd.read_csv("Database.csv")
data = data.dropna()
data.tail()


y = data[data.columns[data.columns=='interest_Rate']]

data = data[data.columns[data.columns !='interest_Rate'] ]
data = data[data.columns[data.columns !='Month'] ]
x = data[data.columns[data.columns !='Year'] ]



#x = data['BusinessApplications','ConstructionSpending', 'BusinessApplications', 'ConstructionSpending' , 'DurableGoodsNewOrders', 'InternationalTrade_Exports' ,  'InternationalTrade_Imports' , 'ManuInventories' , 'ManuNewOrders' , 'NewHomesForSale' , 'NewHomesSold', 'ResConstPermits' , 'ResConstUnitsCompleted, ResConstUnitsStarted' , 'RetailInventories' , 'SalesForRetailAndFood' , 'WholesaleInventories' , 'consumerIndex' , 'UNRATE']
y.head()


lm1 = lm.fit(x, y)
#array([2, 3])
# converts array to list
coef = lm1.coef_.ravel().tolist()
print(coef)

print(lm1.intercept_)
#define the objective function
# prints number of coeficienttes
print(len(coef))
def fitness_function(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    x8 = x[7]
    x9 = x[8]
    x10 = x[9]
    x11 = x[10]
    x12 = x[11]
    x13 = x[12]
    x14 = x[13]
    x15 = x[14]
    x16 = x[15]
    x17 = x[16]
    
    #apply constraints

    penalty = 0
    
    if  -2.680e-06*x1 + -0.00001014*x2 + 0.00001686*x3 + 0.000005249*x4 + -0.00002568*x5 + -0.000002184*x6 + 0.000004081*x7 + 0.018*x8 + 0.003076*x9 + -0.001368*x10 + 0.001669*x11 + -0.0004439*x12 + 0.00001685*x13 + 0.000002279*x14 + -0.0000172*x15 + 0.1582*x16 + -0.09859*x17 > 0.05:
        penalty = np.inf
        
    return - (-2.680e-06*x1 + -0.00001014*x2 + 0.00001686*x3 + 0.000005249*x4 + -0.00002568*x5 + -0.000002184*x6 + 0.000004081*x7 + 0.018*x8 + 0.003076*x9 + -0.001368*x10 + 0.001669*x11 + -0.0004439*x12 + 0.00001685*x13 + 0.000002279*x14 + -0.0000172*x15 + 0.1582*x16 + -0.09859*x17) + penalty
    



import numpy

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover


print(coef)



# Inputs of the equation.
#equaion_inputs = [-2.68E-06,	-1.01E-05,	1.69E-05,	5.25E-06,	-2.57E-05,	-2.18E-06,	4.08E-06,	1.80E-02,	3.08E-03,	-1.37E-03,	1.67E-03,	-4.44E-04,	1.69E-05,	2.28E-06,	-1.72E-05,	-2.99E+01,	-9.86E-02]
equation_inputs = coef
    
    # Number of the weights we are looking to optimize.
num_weights = 17

sol_per_pop = 150
num_parents_mating = 2


# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
new_population = numpy.random.uniform(low=-1.37E-03
, high=1.80E-02
, size=pop_size)
print(new_population)

num_generations = 1000
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measing the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(equation_inputs, new_population)

    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, 
                                      num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    # The best result in the current iteration.
    print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = cal_pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best coefficients : ", new_population[best_match_idx, :])
print("Best Interest rate : ", fitness[best_match_idx] * 100 )


import matplotlib.pyplot as plt
plt.plot(fitness)

plt.title("Best interest rate over (230 months)")
plt.suptitle("Octuber 1994 - Sept 2023 ")
plt.xlabel("Generations Selected") 
plt.ylabel("Interest Rate") 

plt.show()


import numpy as np
from geneticalgorithm import geneticalgorithm as ga


#define the fitness the equations

def fitness_function(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    x8 = x[7]
    x9 = x[8]
    x10 = x[9]
    x11 = x[10]
    x12 = x[11]
    x13 = x[12]
    x14 = x[13]
    x15 = x[14]
    x16 = x[15]
    x17 = x[16]
    
    #apply constraints

    penalty = 0

    if  -2.680e-06*x1 + -0.00001014*x2 + 0.00001686*x3 + 0.000005249*x4 + -0.00002568*x5 + -0.000002184*x6 + 0.000004081*x7 + 0.018*x8 + 0.003076*x9 + -0.001368*x10 + 0.001669*x11 + -0.0004439*x12 + 0.00001685*x13 + 0.000002279*x14 + -0.0000172*x15 + 0.1582*x16 + -0.09859*x17 > .06 or -2.680e-06*x1 + -0.00001014*x2 + 0.00001686*x3 + 0.000005249*x4 + -0.00002568*x5 + -0.000002184*x6 + 0.000004081*x7 + 0.018*x8 + 0.003076*x9 + -0.001368*x10 + 0.001669*x11 + -0.0004439*x12 + 0.00001685*x13 + 0.000002279*x14 + -0.0000172*x15 + 0.1582*x16 + -0.09859*x17 < 0:
        penalty = np.inf
        
    return - (-2.680e-06*x1 + -0.00001014*x2 + 0.00001686*x3 + 0.000005249*x4 + -0.00002568*x5 + -0.000002184*x6 + 0.000004081*x7 + 0.018*x8 + 0.003076*x9 + -0.001368*x10 + 0.001669*x11 + -0.0004439*x12 + 0.00001685*x13 + 0.000002279*x14 + -0.0000172*x15 + 0.1582*x16 + -0.09859*x17) + penalty + lm1.intercept_
    

    ## negative because genetic algoritim will always max and we want to min

# create an instance of the GA solver

algorithm_params = {'max_num_iteration': 50,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model = ga(function = fitness_function, dimension = 17, variable_type = 'bool', algorithm_parameters = algorithm_params )


model.run()




x1 = data[data.columns[data.columns == 'UNRATE'] ]


lm5 = lm.fit(x1,y)



print(lm5.intercept_)
print(lm5.coef_)












# When we thing that Fed will stop increasing the interest rate

def fitness(x):
    x1 = x[0]
    
    #apply constraints

    penalty = 0
    
    if  lm5.coef_*x1   > 0.03 or lm5.coef_*x1   < 0.00 :
        penalty = np.inf
        
    return  (lm5.coef_*x1 ) + penalty + lm5.intercept_
  
  # create an instance of the GA solver


algorithm_params = {'max_num_iteration': 50,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}


modelInf = ga(function = fitness, dimension = 1, variable_type = 'bool', algorithm_parameters = algorithm_params )

modelInf.run()