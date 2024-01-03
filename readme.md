Code for an unpublished research paper.

# Concrete Cost Estimates
This modifies the code from (https://github.com/jschanck/eprint-2019-1161/), which estimates the cost for both quantum and classical lattice sieves, to include memory costs and recursive strategies in the `list_decoding` subroutine.

To run the estimates, from this directory call:

```
python3 runall.py
```

This will compute the results and save them as csv files in a `data` directory. These are the results that appear in Tables 1 and 2 of the paper.

There are several arguments one can give to `runall.py`:

- `--ds`: This is a list of dimensions for the sieve. It defaults to the dimensions listed in the Kyber and Dilithium specifications for the primal attacks.
- `--depths`: This is a list of possible recursion depths to estimate for the sieve. It defaults to {1,2,3}, where 1 is equivalent to the previous [BDGL16](https://eprint.iacr.org/2015/1128) approach, albeit with memory costs
- `--metric`: This is the cost metric to use (or a list of them), which can either be `hardware_time`, which will add memory costs, or `classical`, which does not add memory costs
- `--mem_cost`: This supplies the constant C for memory access, so that routing N bits of data has cost C * N^{1+Delta} for an argument Delta. This can also be a list of such costs. It defauls to `2.0**(-19.8)`.
- `--mem_exp`: This supples the exponent Delta such that routing N bits of data has cost C * N^{1+Delta}. It defaults to 0.5
- `--approximate`: If this is included, only an approximate cost will be calculated (see below), otherwise an exact cost will be computed.
- `--jobs`: Gives the number of threads to use.

To roughly summarize the logic, the list decoding algorithm has several parameters to use, most importantly being the filter angle theta and the code size. Both of these must be chosen for all levels of recursion, resulting in a parameter space that grows exponentially with recursion size. Thus, for large recursion depths, it is computationally intensive to find optimal values. Moreover, several key probabilities and lattice properties require computationally intensive integrals to compute properly. 

Thus, the logic of this code is to approximate most of the hard-to-compute probabilities in order to quickly find optimal parameters for the algorithm, then only compute the probabilities/etc. afterwards.

One can compute this in two steps by including the `--approximate` command line argument, which tells the script to only compute the approximate results and save them to disk. If `--approximate` is not included, the script will compute an exact cost, and it will search on disk for the parameters from a previous approximate computation. If it finds no such results, it will print a warning and then compute the approximate result itself.

# Asymptotic Cost Estimates

Calling

```
python3 recursive_costs.py
```
will iterate through various values of the memory exponent `Delta` and for various recursion depths, find an optimal `alpha` argument for each, and give the exponent for the cost to sieve. Currently it outputs the results as space-separated values which are used to generate Figure 1 from the table. Alternatively, one can un-comment the code to directly plot the result with `matplotlib`.

# Requirements

Tested on Python 3.8.10. To install required modules:

```
pip installl -r requirements.txt
```

# Differences from eprint-2019-1161

## cost.py

Most of the differences are here. The main difference is the explicit inclusion of memory costs, as well as the ability to compute costs recursively. Specific changes are documented in the code itself. Besides this:

- it is switched from a "vector-first" approach to "bucket-first" approach.

- the code size and argument theta are optimized via a direct binary search based on properties of the cost function, rather than relying on a scipy optimizer library

- the number of codes to test includes a concentration bound to ensure enough solutions are found

## probabilities.py

Many functions are now given an argument `kappangle`, which replaces previously hardcoded values of `mp.pi / 3`. The reason is that for a single level of the BDGL or BGL sieve, we only want vectors at an angle at most `mp.pi / 3`. However, when recursing, the acceptable angle increases. 

Possibly because of this, the probabilities given by `pf` and `ngr` often gave results with errors so large that the resulting values were meaningless. Thus, they were changed to subdivide the interval of the integral until the errors are sufficiently small. This is extremely computationally intensive, so this will parallelize when allowed.

There is also a new function `p_recursion_hit` which computes the probability that two vectors which are reducing according to some angle will be found by the recursive step in the sieve (see appendix of the paper)

## exact.py and approximate.py
This are new files of wrapper functions to compute many exact (resp. approximate) `list_decoding` costs

## config.py

This contains a number of new magic constants. Specifically:

- `SORT_EXPONENT_DELTA`: This is the exponent `Delta` from the paper. This will be modified globally by `runall.py`; it is better to change the command-line argument to `runall.py`, rather than modify the line in `config.py`
- `BIT_OPS_PER_SORT_BIT`: This is the constant C for the sort cost. Again, this is modified globally by `runall.py`
- `COLLISION_BUFFER`: When sorting vectors into their respective buckets, I imagine a hashmap data structure where vectors are sorted based on the hash of the codeword. This means we do not need to store the entire codeword when sorting the vectors, just a hash of it, which reduces memory requirement. To ensure that there are no collisions in these hashes, I assume the output of the hash has 2lg(N)+`COLLISION_BUFFER` bits, where N is the number of codewords.
- `SORT_CONSTANT`: The code includes a minimum RAM model cost of a sort, which is SORT_CONSTANT * N * log(N,2) for N bits.
- `PROB_MIN`: `list_decoding_internal` decides on the number of codes to try to catch all reducing vectors. We can use a concentration bound on a binomial distribution to estimate how many codes it must try, but this requires us to have some probability that we do not recover enough codes (i.e., that the sieve fails). That probability is what `PROB_MIN` indicates.

