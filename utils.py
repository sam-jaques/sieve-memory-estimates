# -*- coding: utf-8 -*-
"""
Highlevel functions for bulk producing estimates.
"""
from mpmath import mp
from collections import OrderedDict
from probabilities import probabilities, Probabilities
from config import MagicConstants

import os
import csv
try:
    import cPickle as pickle
except ImportError:
    import pickle


class PrecomputationRequired(Exception):
    pass


def pretty_probs(probs, dps=10):
    """
    Take a ``Probabilities`` object and pretty print the estimated probabilities.

    :param probs: a ``Probabilitiess`` object.

    """
    fmt = "{0:7s}: {1:%ds}" % dps
    with mp.workdps(dps):
        print(fmt.format("gr", probs.gr))
        print(fmt.format("ngr", 1 - probs.gr))
        print(fmt.format("pf", probs.pf))
        print(fmt.format("npf", 1 - probs.pf))
        print(fmt.format("gr^pf", probs.gr_pf))
        print(fmt.format("ngr^pf", probs.ngr_pf))
        print(fmt.format("gr|pf", probs.gr_pf / probs.pf))
        print(fmt.format("pf|gr", probs.gr_pf / probs.gr))
        print(fmt.format("ngr|pf", probs.ngr_pf / probs.pf))


def create_bundle(d, n, K=None, BETA=None, prec=None):
    """
    Create a bundle of probabilities.

    :param d: We consider the sphere `S^{d-1}`.
    :param n: Number of popcount vectors.
    :param K: We consider all `k ∈ K` as popcount thresholds (default `k = 5/16⋅n`).
    :param BETA: We consider all caps parameterized by `β in BETA` (default: No cap).
    :param prec: We compute with this precision (default: 53).

    """
    bundle = OrderedDict()

    prec = prec if prec else mp.prec
    BETA = BETA if BETA else (None,)
    K = K if K else (int(MagicConstants.k_div_n * n),)

    # if 2 ** mp.floor(mp.log(n, 2)) != n:
    #     raise ValueError("n must be a power of two but got %d" % n)

    for k in K:
        if not 0 <= k <= n:
            raise ValueError("k not in [0, %d]" % (0, n))

    for beta in BETA:
        beta_mpf = mp.mpf(beta) if beta else None
        beta_flt = float(beta) if beta else None
        for k in K:
            bundle[(d, n, k, beta_flt)] = probabilities(d, n, k, beta=beta_mpf, prec=prec)

    return bundle


def bundle_fn(d, n=None):
    if n is None:
        d, n = [keys[:2] for keys in d.keys()][0]
    return os.path.join("probabilities", "%03d_%04d" % (d, n))


def store_bundle(bundle):
    """
    Store a bundle in a flat format for compatibility reasons.

    In particular, mpf values are converted to strings.

    """
    bundle_ = OrderedDict()

    for (d, n, k, beta) in bundle:
        with mp.workprec(bundle[(d, n, k, beta)].prec):
            vals = OrderedDict([(k_, str(v_)) for k_, v_ in bundle[(d, n, k, beta)]._asdict().items()])
        bundle_[(d, n, k, beta)] = vals

    with open(bundle_fn(bundle), "wb") as fh:
        pickle.dump(bundle_, fh)


def load_bundle(d, n, compute=False):
    """
    Load bundle from the flat format and convert into something we can use.

    """
    bundle = OrderedDict()
    try:
        with open(bundle_fn(d, n), "rb") as fh:
            bundle_ = pickle.load(fh)
            for (d, n, k, beta) in bundle_:
                with mp.workprec(int(bundle_[(d, n, k, beta)]["prec"])):
                    d_ = dict()
                    for k_, v_ in bundle_[(d, n, k, beta)].items():
                        if "." in v_:
                            v_ = mp.mpf(v_)
                        elif v_ == "None":
                            v_ = None
                        else:
                            v_ = int(v_)
                        d_[k_] = v_
                    bundle[(d, n, k, beta)] = Probabilities(**d_)
            return bundle
    except IOError:
        if compute:
            bundle = create_bundle(d, n, prec=int(compute))
            store_bundle(bundle)
            return bundle
        else:
            raise PrecomputationRequired("d: {d}, n: {n}".format(d=d, n=n))


def __bulk_create_and_store_bundles(args):
    d, n, BETA, prec = args
    bundle = create_bundle(d, n, BETA=BETA, prec=prec)
    store_bundle(bundle)


def bulk_create_and_store_bundles(
    D,
    N=(128, 256, 512, 1024, 2048, 4096, 8192),
    BETA=(None, mp.pi / 3 - mp.pi / 10, mp.pi / 3, mp.pi / 3 + mp.pi / 10),
    prec=2 * 53,
    ncores=1,
):
    """
    Precompute a bunch of probabilities.
    """
    from multiprocessing import Pool

    jobs = []
    for d in D:
        for n in N:
            jobs.append((d, n, BETA, prec))

    if ncores > 1:
        return list(Pool(ncores).imap_unordered(__bulk_create_and_store_bundles, jobs))
    else:
        return map(__bulk_create_and_store_bundles, jobs)


def load_probabilities(d, n, k, beta=None, compute=False):
    probs = load_bundle(d, n, compute=compute)[(d, n, k, beta)]
    return probs


def __bulk_cost_estimate(args):
    try:
        f, d, metric, kwds = args
        return f(d, metric=metric, **kwds)
    except Exception as e:
        print("Exception in f: {f}, d: {d}, metric: {metric}".format(f=f, d=d, metric=metric))
        raise e


def new_bulk_cost_estimate(mem_cost, mem_exponent, depths, D, metric, filename = None, exact=True, ncores = 1, exhaustive_size = 0, **kwds):
    """
    Run cost estimates and write to csv file.

    :param mem_cost: a constant for how much a sort should cost or an iterable of those
    :param depths: an iterable of recursion depths to estimate
    :param D: an iterable of dimensions to run ``list_decoding`` on
    :param metric: a metric from ``Metrics`` or an iterable of such metrics
    :param filename: csv filename to write to (must accept "{exact}" placeholder)
    :param exact: if true, computes exact results; if false, computes approximate results
    :param ncores: number of CPU cores to use
    :returns: ``None``, but files are written to disk.

    """
    from cost import LogicalCosts, ClassicalCosts, ClassicalMetrics, SizeMetrics

    
    try:
        for mem_cost_ in mem_cost:
            new_bulk_cost_estimate(mem_cost_, mem_exponent, depths, D, metric, exact=exact, ncores = ncores,exhaustive_size = exhaustive_size, **kwds)
        return
    except TypeError:
        pass

    try:
        for mem_exponent_ in mem_exponent:
            new_bulk_cost_estimate(mem_cost, mem_exponent_, depths, D, metric, exact=exact, ncores = ncores,exhaustive_size = exhaustive_size, **kwds)
        return
    except TypeError:
        pass

    if not isinstance(metric, str):
        for metric_ in metric:
            new_bulk_cost_estimate(mem_cost, mem_exponent,depths, D, metric_, exact=exact, ncores = ncores,exhaustive_size = exhaustive_size, **kwds)
        return

    from config import MultiProcessingConfig, MagicConstants
    MultiProcessingConfig.num_cores = ncores
    MagicConstants.BIT_OPS_PER_SORT_BIT = mem_cost
    MagicConstants.SORT_EXPONENT_DELTA = mem_exponent

    if filename is None:
        filename = os.path.join("data", "{exact}-cost-estimate-{delta}.csv")
    exact_filename = filename.format(exact="exact",delta=mem_exponent)
    approximate_filename = filename.format(exact="approximate",delta=mem_exponent)

    if exact:
        from exact import compute_exact_costs
        compute_exact_costs(D, depths, mem_cost, metric, exact_filename, approximate_filename)
    else:
        from approximate import compute_approximate_costs
        compute_approximate_costs(D, depths, mem_cost, metric, approximate_filename,exhaustive_size)


def bulk_cost_estimate(f, D, metric, filename=None, ncores=1, **kwds):
    """
    Run cost estimates and write to csv file.

    :param f: one of ``all_pairs``, ``random_buckets`` or ``list_decoding`` or an iterable of those
    :param D: an iterable of dimensions to run ``f`` on
    :param metric: a metric from ``Metrics`` or an iterable of such metrics
    :param filename: csv filename to write to (may accept "{metric}" and "{f}" placeholders)
    :param ncores: number of CPU cores to use
    :returns: ``None``, but files are written to disk.

    """
    from cost import LogicalCosts, ClassicalCosts, QuantumMetrics, ClassicalMetrics, SizeMetrics

    try:
        for f_ in f:
            bulk_cost_estimate(f_, D, metric, ncores=ncores, **kwds)
        return
    except TypeError:
        pass

    if not isinstance(metric, str):
        for metric_ in metric:
            bulk_cost_estimate(f, D, metric_, ncores=ncores, **kwds)
        return



    from multiprocessing import Pool

    jobs = []
    for d in D[::-1]:
        jobs.append((f, d, metric, kwds))

    if ncores > 1:
        r = list(Pool(ncores).imap_unordered(__bulk_cost_estimate, jobs))
    else:
        r = list(map(__bulk_cost_estimate, jobs))

    r = sorted(r)  # relying on "d" being the first entry here

    if filename is None:
        filename = os.path.join("data", "cost-estimate-{f}-{metric}.csv")

    filename = filename.format(f=f.__name__, metric=metric)

    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        fields = r[0]._fields[:-1]
        if r[0].metric in QuantumMetrics:
            fields += LogicalCosts._fields[1:]
        elif r[0].metric in ClassicalMetrics:
            fields += ClassicalCosts._fields[1:]
        elif r[0].metric in SizeMetrics:
            pass
        else:
            raise ValueError("Unknown metric {metric}".format(metric=r[0].metric))
        csvwriter.writerow(fields)

        for r_ in r:
            csvwriter.writerow(r_[:-1] + r_.detailed_costs[1:])
