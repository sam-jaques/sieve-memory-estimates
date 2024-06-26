from cost import *
from config import MagicConstants, MultiProcessingConfig
from multiprocessing import Pool,get_context
import csv
import os.path
from probabilities import DisplayConfig


# Function for a single thread
# args is expected to be a list of tuples
# (dimension, recursion depth, metric)
def get_approximate_costs(args):
    results = []
    for arg in args:
        d,depth,metric, exhaustive_size = arg
        n = 1
        while n < d:
            n = 2 * n
        n = n-1
        k = int(MagicConstants.k_div_n * (n))
        results.append(list_decoding(
            d=d,
            n=[n]*depth, 
            k = [k]*depth, 
            theta = None, 
            given_code_size = None, 
            optimize=False,
            fast = True, 
            metric=metric,
            allow_suboptimal=False,
            recursion_depth = depth, 
            exhaustive_size = exhaustive_size))
    return results


def compute_approximate_costs(ds, recursion_depths, mem_cost, metric, filename,exhaustive_size = 0):
    output_file = open(filename,'a')
    writer = csv.writer(output_file)
    # This re-writes the headers; this is somewhat desirable
    # since the headers can change based on recursion depth
    writer.writerow(list_decoding_title(max(recursion_depths)))
    # Limits number of threads based on the total number of jobs
    ncores = min(MultiProcessingConfig.num_cores,len(ds)*len(recursion_depths))
    if ncores > 1:
        DisplayConfig.display = False
    jobs = [(d, depth, metric,exhaustive_size) for d in ds for depth in recursion_depths]
    # This splits the jobs up as evenly as it can
    # However, larger recursion depths are much longer-running
    # so typically there will be 1 or 2 threads that take a long
    # time to finish. A more efficient parallelization would be 
    # to have a stack of jobs and each thread would retrieve a 
    # new one once it finishes a previous one. Todo.
    sub_jobs = [jobs[i::ncores] for i in range(ncores)]
    if ncores > 1:
        with get_context("fork").Pool(processes=ncores) as pool:
            results = pool.map(get_approximate_costs,sub_jobs)
    else:
        results = [get_approximate_costs([job]) for job in jobs]
    for result in results: # by processor
        for cost in result:
            writer.writerow(list_decoding_as_list(cost))
    output_file.close()
