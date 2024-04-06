import math


# Computes the asymptotic cost of solving SphereFind
# with the SphereFilter algorithm
'''
 	n: the list size to the power of 2/d
 	kappa: the required angle to find
 	alpha_sq: code parameter alpha^2
 	Delta: physical connectivity
 	recursion_depth: Number of depths of recursion remaining

 	Returns the cost to the power of 2/d
'''
def sphere_find_cost(n,kappa,alpha_sq,Delta,recursion_depth):
	Na = 1/(1-alpha_sq)
	Nt = ((1-alpha_sq)**2)/((1-kappa)*(1+kappa-2*alpha_sq))
	expected_solutions = (1-kappa**2)*(n**2)
	bucket_size = max(1,n*(1-alpha_sq))
	code_size = Na
	num_codes = max(1,expected_solutions*Nt/(code_size*(n**2 * (1-alpha_sq)**2)))
	sort_list_cost = pow(n,1+Delta)
	sort_buckets_cost = pow(bucket_size*code_size,1+Delta)
	if recursion_depth == 1:
		search_buckets_cost = code_size * bucket_size**2
	else:
		sub_kappa =  (kappa - alpha_sq)/(1-alpha_sq)
		search_buckets_cost = code_size * sphere_find_cost(bucket_size,sub_kappa,alpha_sq,Delta,recursion_depth-1)
	return num_codes* max(sort_list_cost, sort_buckets_cost, search_buckets_cost)


# Find the optimal alpha for a given SphereFind
# problem
def optimal_alpha(n,kappa,Delta,recursion_depth):
	low_alpha = 1e-5
	high_alpha = 0.5 - 1e-5
	while abs(high_alpha - low_alpha) > 1e-5:
		alpha_sq = (high_alpha + low_alpha)/2
		cost1 = sphere_find_cost(n,kappa,alpha_sq,Delta,recursion_depth)
		cost2 = sphere_find_cost(n,kappa,alpha_sq+1e-6,Delta,recursion_depth)
		if cost1 < cost2:
			high_alpha = alpha_sq
		else:
			low_alpha = alpha_sq
	return (high_alpha+low_alpha)/2

# Finds the optimal alpha for spherefind
# and then computes the cost
# as the leading exponent C in 2^{Cd + o(d)}
def best_sieve_cost(n,kappa,Delta,recursion_depth):
	alpha_sq = optimal_alpha(n,kappa,Delta,recursion_depth)
	return math.log(sphere_find_cost(n,kappa,alpha_sq,Delta,recursion_depth))/(2*math.log(2))


import numpy as np
import matplotlib.pyplot as plt


# Basic lattice sieving problem
n = 4/3
kappa = 0.5

costs = []
delta_values = list(np.linspace(0, 0.4094, 90))+list(np.linspace(0.4094,0.5,10))
recursion_depths = [1,2,3,4,8,16,32,64,128]
for d in recursion_depths:
	costs += [[best_sieve_cost(n,kappa,delta,d) for delta in delta_values]]

## Plots the values directly
# for d in range(len(recursion_depths)):
# 	plt.plot(delta_values, costs[d],label=recursion_depths[d])

# # # plt.plot(delta_values, diff_values, label='Diff')
# plt.xlabel('Connectivity parameter Delta')
# plt.ylabel('Exponent in cost function')
# plt.title('Recursive Sieving Costs')
# plt.legend()
# plt.grid()
# plt.show()


## Outputs to a latex format
with  open("recursion-plot.csv",'w') as file:
	file.write("delta    ")
	for depth in recursion_depths:
		file.write("    " + str(depth))
	file.write("\n")
	for ii in range(len(delta_values)):
		file.write(str(delta_values[ii]))
		for i in range(len(recursion_depths)):
			file.write("    "+str(costs[i][ii]))
		file.write("\n")
