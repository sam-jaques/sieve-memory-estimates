from cost import list_decoding, list_decoding_internal, Metrics, list_decoding_title, list_decoding_as_list
from config import MagicConstants
import csv
import os.path
from multiprocessing import Pool

from probabilities import ngr, pf, MultiProcessingConfig, DisplayConfig

MultiProcessingConfig.num_cores = 11

approx_filename = "approximate_results.csv"
exact_filename = "exact_results.csv"
approximate_costs = {}
if os.path.isfile(approx_filename):
	with open(approx_filename,'r') as approx_file:
		reader = csv.DictReader(approx_file)
		for row in reader:
			ns = []
			ks = []
			thetas = []
			codes = []
			for i in range(1,int(row['recursion_depth'])+1):
				ns.append(row['n_'+str(i)])
				ks.append(row['k_'+str(i)])
				thetas.append(row['theta_'+str(i)])
				codes.append(row['code_size_'+str(i)])
			approximate_costs[(row['dimension'],row['metric'],row['memory_cost'],row['recursion_depth'])] = {"ns":ns,"ks":ks,"thetas":thetas,"codes":codes}




recursion_depths = [1,2,3,4]
ds =  [375, 586, 829, 394, 587, 818]

mem_costs = [2.0**(-19.8), 1]

# approximate_costs = 

def get_exact_cost(args):
	d, metric, mem_cost, recursion_depth = args
	ns = None
	ks = None
	thetas = None
	codes = None
	optimize = True
	if args in approximate_costs:
		ns = approximate_costs[args]["ns"]
		ks = approximate_costs[args]["ks"]
		thetas = approximate_costs[args]["thetas"]
		codes = approximate_costs[args]["codes"]
		optimize = False
	return list_decoding(
	    d=d, 
	    n=ns, 
	    k=ks, 
	    theta=thetas, 
	    given_code_size=codes, 
	    optimize=optimize, 
	    metric=metric, 
	    recursion_depth = recursion_depth
	)

def get_many_exact_costs(args):
	ds, depths, metric = args
	mem_cost = MagicConstants.BIT_OPS_PER_SORT_BIT
	results = []
	for depth in depths:
		for d in ds:
			results.append(get_exact_cost((d,metric,mem_cost,depth)))
	return results

# Currently the multi-processing is useless because 
# if we have the approximate results, the majority cost
# is in the probability integrals, which are *already*
# heavily parallelized

DisplayConfig.display = True

max_recursion_depth = 1
recursion_depths = [i+1 for i in range(max_recursion_depth)]
ds =  [375, 586, 829, 394, 587, 818]
mem_costs = [2.0**(-19.8), 1]

output_file = open(exact_filename,'a')
writer = csv.writer(output_file)
# This re-writes the headers; this is somewhat desirable
# since the headers can change based on recursion depth
writer.writerow(list_decoding_title(max_recursion_depth))
# Is it better to parellelize in the probabilities?
ncores = 1 # min(MultiProcessingConfig.num_cores,len(ds))
for metric in ["hardware_time","classical"]:
    if metric == "classical":
        sub_mem_costs = [1]
    else:
        sub_mem_costs = mem_costs

    sub_ds = [ds[i::ncores] for i in range(ncores)]
    jobs = [(sub_d, recursion_depths, metric) for sub_d in sub_ds]
    for mem_cost in mem_costs:
        MagicConstants.BIT_OPS_PER_SORT_BIT = mem_cost
        # Create a Pool with the specified number of cores
        if ncores > 1:
            with Pool(processes=ncores) as pool:
                # Apply the parallel function to each chunk using the Pool
                results = pool.map(get_many_exact_costs,jobs)
        else:
            results = [get_many_exact_costs(job) for job in jobs]
    for result in results: # by processor
        for cost in result:
            writer.writerow(list_decoding_as_list(cost))

output_file.close()


# print(pf(375, 511, 170, beta=1.1780892556129021, lb=None, ub=None, beta_and=False, integrate = True, prec=None))
# print(ngr(375, 1.1780892556129021))