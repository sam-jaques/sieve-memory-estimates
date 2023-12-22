from cost import *
from config import MagicConstants, MultiProcessingConfig
from multiprocessing import Pool
import csv
import os.path
from probabilities import DisplayConfig

max_recursion_depth = 4
recursion_depths = [4]
ds =  [375, 586, 829, 394, 587, 818]
mem_costs = [2.0**(-19.8), 1]




MultiProcessingConfig.num_cores = 11

# Because the memory cost is a global constant
# we only multi-thread for different recursion depths
# and dimensions
def get_approximate_costs(args):
    ds,depths,metric = args
    results = []
    for depth in depths:
        for d in ds:
            n = 1
            while n < d:
                n = 2 * n
            n = n-1
            k = int(MagicConstants.k_div_n * (n))
            results.append(list_decoding_internal(
                d=d,
                n_in=[n]*depth, 
                k_in = [k]*depth, 
                theta_in = None, 
                given_code_size = None, 
                optimize = True, 
                metric=metric,
                allow_suboptimal=False,
                recursion_depth = depth, 
                exact = True))
    return results


DisplayConfig.display = False

filename = 'approximate_results.csv'

output_file = open(filename,'a')
writer = csv.writer(output_file)
# This re-writes the headers; this is somewhat desirable
# since the headers can change based on recursion depth
writer.writerow(list_decoding_title(max_recursion_depth))
for metric in ["hardware_time"]:#,"classical"]:
    if metric == "classical":
        sub_mem_costs = [1]
    else:
        sub_mem_costs = mem_costs
    ncores = min(MultiProcessingConfig.num_cores,len(ds))
    sub_ds = [ds[i::ncores] for i in range(ncores)]
    jobs = [(sub_d, recursion_depths, metric) for sub_d in sub_ds]
    for mem_cost in mem_costs:
        MagicConstants.BIT_OPS_PER_SORT_BIT = mem_cost

        if ncores > 1:
            with Pool(processes=ncores) as pool:
                # Apply the parallel function to each chunk using the Pool
                results = pool.map(get_approximate_costs,jobs)
        else:
            results = [get_approximate_costs(job) for job in jobs]
    for result in results: # by processor
        for cost in result:
            writer.writerow(list_decoding_as_list(cost))

output_file.close()
