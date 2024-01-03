from cost import list_decoding, list_decoding_internal, Metrics, list_decoding_title, list_decoding_as_list
from config import MagicConstants
import csv
import os.path
import multiprocessing as mp

from probabilities import ngr, pf, MultiProcessingConfig, DisplayConfig


approximate_costs = {}

# Loads approximate costs from an existing file
def load_approximate_costs(approx_filename):
    if os.path.isfile(approx_filename):
        with open(approx_filename,'r') as approx_file:
            reader = csv.DictReader(approx_file)
            for row in reader:
                # Catches extra header rows
                if row['dimension'] == "dimension":
                    continue
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


# Finds a single exact cost
# It tries to use a result from an approximate cost, if it exists
# to save computation of the optimal arguments (theta, code_size, etc.)
def get_exact_cost(args):
    d, metric, mem_cost, recursion_depth = args
    ns = None
    ks = None
    thetas = None
    codes = None
    optimize = True
    key = (str(d),metric,"{:.7f}".format(mem_cost),str(recursion_depth)) 
    if key in approximate_costs:
        ns = None
        ks = None
        thetas = [float(theta) for theta in approximate_costs[key]["thetas"]]
        codes = [2**float(code) for code in approximate_costs[key]["codes"]]
        optimize = False
    else:
        print("Warning: not all approximate costs pre-computed")

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



# Iterates over arguments to compute exact costs
# If queue is None it will attempt to write directly to
# disk, otherwise it will send the results to `queue`
# for some other thread to save them
def get_many_exact_costs(args, queue, filename = None):
    mem_cost = MagicConstants.BIT_OPS_PER_SORT_BIT
    for d,depth,metric in args:
        result = get_exact_cost((d,metric,mem_cost,depth))
        if queue:
            queue.put(result)
        else:
            with open(filename, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(list_decoding_as_list(result))

def listener(filename, queue):
    with open(filename, 'a') as file:
        writer = csv.writer(file)
        while True:
            message = queue.get()
            if message == 'kill':
                break
            writer.writerow(list_decoding_as_list(message))




# This can parallelize or not;
# I found it worked better to parallelize within the probability computations
# rather than run multiple computations at once
# So the default option is not to parallelize at all
def compute_exact_costs(ds, recursion_depths, mem_cost, metric, exact_filename, approximate_filename, parallelize=False):
    load_approximate_costs(approximate_filename)
    # This re-writes the headers; this is somewhat desirable
    # since the headers can change based on recursion depth
    output_file = open(exact_filename,'a')
    writer = csv.writer(output_file)
    writer.writerow(list_decoding_title(max(recursion_depths)))
    output_file.close()
    all_args = [(d,depth,metric) for d in ds for depth in recursion_depths]
    ncores =  min(MultiProcessingConfig.num_cores,len(ds)*len(recursion_depths))
    if not parallelize or ncores == 1:
        get_many_exact_costs(all_args, None, exact_filename)
    else:
        jobs = []
        for i in range(ncores):
            jobs.append(all_args[i::ncores])
        ncores = 1
        # if ncores > 1:
        manager = mp.Manager()
        queue = manager.Queue()
        # we've "taken" the cores, can't let anything else use them
        MultiProcessingConfig.num_cores = 1
        with mp.Pool(processes=ncores) as pool:
            workers = []
            # Write to the file
            watcher = pool.apply_async(listener, (exact_filename,queue))
            # Apply the parallel function to each chunk using the Pool
            for job in jobs:
                workers.append(pool.apply_async(get_many_exact_costs, (job, queue)))
            # results = pool.map(get_many_exact_costs,jobs_w_queue)
            for worker in workers:
                worker.get()
                queue.put('kill')

