import numpy
import ga


p_naut=3


num_coeff=4

"""
    Mating pool size
    Population size
"""
sol_per_pop = 3
num_parents_mating =2

# Defining the population size.
pop_size = (sol_per_pop,num_coeff) # The population will have sol_per_pop chromosome where each chromosome has num_coeff genes.

#Creating the initial population.
matrix_A = numpy.random.uniform(low=1, high=1000, size=(sol_per_pop,1))
matrix_a = numpy.random.uniform(low=1, high=500, size=(sol_per_pop,1))
matrix_B = numpy.random.uniform(low= 1, high= 300, size=(sol_per_pop,1))
matrix_b = numpy.random.uniform(low= 1
, high= 500, size=(sol_per_pop,1))
for i in range(matrix_B.shape[0]):
	if matrix_B[i,0]>=matrix_b[i,0]:
		temp=matrix_b[i,0]
		matrix_b[i,0]=matrix_B[i,0]
		matrix_B[i,0]=temp
matrix_b=matrix_b*(-1)
new_population = numpy.concatenate((matrix_A,matrix_a,matrix_B,matrix_b),axis=1)
print(new_population)


num_generations=100
price_previous=None

for generation in range(num_generations):
	print("Generation : ", (generation+1))
	print("Initial generation : \n", (new_population))
	# Measing the fitness of each chromosome in the population.
	fitness= ga.cal_pop_fitness(p_naut,new_population,(generation+1),price_previous)
	# Selecting the best parents in the population for mating.
	parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)
	print("Selected parents to mate : \n", (parents))
	# Generating next generation using crossover.
	offspring_crossover = ga.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_coeff))
	print("After crossover offspring: \n", (offspring_crossover))
	# Adding some variations to the offsrping using mutation.
	offspring_mutation = ga.mutation(offspring_crossover)
	print("After mutation : \n", (offspring_mutation))
	# Creating the new population based on the parents and offspring.
	new_population[0:parents.shape[0], :] = parents
	new_population[parents.shape[0]:, :] = offspring_mutation
	print("New generation : \n", (new_population))
	price_previous=ga.price(p_naut,new_population,generation+1)
	fitness=ga.cal_pop_fitness(p_naut,new_population,generation+2,price_previous)
	max_fitness_idx = numpy.where(fitness == numpy.min(fitness))
	max_fitness_idx = max_fitness_idx[0][0]
	print("Best result : ", new_population[max_fitness_idx, :])
		
# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(p_naut, new_population,5,price_previous)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx] * 100)