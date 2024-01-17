#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantum and Classical Nearest Neighbor Cost.
"""

from mpmath import mp
from collections import namedtuple
from utils import load_probabilities, PrecomputationRequired
from config import MagicConstants
from probabilities import W, Wmatched, C, pf, ngr_pf, ngr, p_recursion_hit, DisplayConfig
from ge19 import estimate_abstract_to_physical
from sys import stdout
from tqdm import tqdm



"""
COSTS
"""

"""
Logical Quantum Costs

:param label: arbitrary label
:param qubits_in: number of input qubits
:param qubits_out: number of output qubits
:param qubits_max:
:param depth: longest path from input to output (including identity gates)
:param gates: gates except identity gates
:param dw: not necessarily depth*qubits
:param toffoli_count: number of Toffoli gates
:param t_count: number of T gates
:param t_depth: T gate depth

"""

LogicalCosts = namedtuple(
    "LogicalCosts",
    (
        "label",
        "qubits_in",
        "qubits_out",
        "qubits_max",  # NOTE : not sure if this is useful
        "depth",
        "gates",
        "dw",
        "toffoli_count",  # NOTE: not sure if this is useful
        "t_count",
        "t_depth",
    ),
)

"""
Classic Costs

:param label: arbitrary label
:param gates: number of gates
:param depth: longest path from input to output

"""

ClassicalCosts = namedtuple("ClassicalCosts", ("label", "gates", "depth"))

"""
METRICS
"""

ClassicalMetrics = {
    "classical",  # gate count
    "naive_classical",  # query cost
    "hardware_time", # includes latency
}

QuantumMetrics = {
    "g",  # gate count
    "dw",  # depth x width
    "ge19",  # depth x width x physical qubit measurements Gidney Ekera
    "t_count",  # number of T-gates
    "naive_quantum",  # query cost
}

Metrics = ClassicalMetrics | QuantumMetrics

SizeMetrics = {"vectors", "bits"}


def log2(x):
    return mp.log(x) / mp.log(2)


def local_min(f, low=None, high=None):
    """
    Search the neighborhood around ``f(x)`` for a local minimum between ``low`` and ``high``.

    :param f: function to call
    :param low: lower bound on input space
    :param high: upper bound on input space

    """
    from scipy.optimize import fminbound

    def ff(x):
        try:
            return float(f(float(x)))
        except AssertionError:
            return mp.mpf("inf")

    return fminbound(ff, float(low), float(high))




def null_costf(qubits_in=0, qubits_out=0):
    """
    Cost of initialization/measurement.
    """

    return LogicalCosts(
        label="null",
        qubits_in=qubits_in,
        qubits_out=qubits_out,
        qubits_max=max(qubits_in, qubits_out),
        gates=0,
        depth=0,
        dw=0,
        toffoli_count=0,
        t_count=0,
        t_depth=0,
    )


def delay(cost, depth, label="_"):
    # delay only affects the dw cost
    dw = cost.dw + cost.qubits_out * depth
    return LogicalCosts(
        label=label,
        qubits_in=cost.qubits_in,
        qubits_out=cost.qubits_out,
        qubits_max=cost.qubits_max,
        gates=cost.gates,
        depth=cost.depth + depth,
        dw=dw,
        toffoli_count=cost.toffoli_count,
        t_count=cost.t_count,
        t_depth=cost.t_depth,
    )


def reverse(cost):
    return LogicalCosts(
        label=cost.label,
        qubits_in=cost.qubits_out,
        qubits_out=cost.qubits_in,
        qubits_max=cost.qubits_max,
        gates=cost.gates,
        depth=cost.depth,
        dw=cost.dw,
        toffoli_count=cost.toffoli_count,
        t_count=cost.t_count,
        t_depth=cost.t_depth,
    )


def compose_k_sequential(cost, times, label="_"):
    # Ensure that sequential composition makes sense
    assert cost.qubits_in == cost.qubits_out

    return LogicalCosts(
        label=label,
        qubits_in=cost.qubits_in,
        qubits_out=cost.qubits_out,
        qubits_max=cost.qubits_max,
        gates=cost.gates * times,
        depth=cost.depth * times,
        dw=cost.dw * times,
        toffoli_count=cost.toffoli_count * times,
        t_count=cost.t_count * times,
        t_depth=cost.t_depth * times,
    )


def compose_k_parallel(cost, times, label="_"):
    return LogicalCosts(
        label=label,
        qubits_in=times * cost.qubits_in,
        qubits_out=times * cost.qubits_out,
        qubits_max=times * cost.qubits_max,
        gates=times * cost.gates,
        depth=cost.depth,
        dw=times * cost.dw,
        toffoli_count=times * cost.toffoli_count,
        t_count=times * cost.t_count,
        t_depth=cost.t_depth,
    )


def compose_sequential(cost1, cost2, label="_"):
    # Ensure that sequential composition makes sense
    assert cost1.qubits_out >= cost2.qubits_in

    # Pad unused wires with identity gates
    dw = cost1.dw + cost2.dw
    if cost1.qubits_out > cost2.qubits_in:
        dw += (cost1.qubits_out - cost2.qubits_in) * cost2.depth
    qubits_out = cost1.qubits_out - cost2.qubits_in + cost2.qubits_out
    qubits_max = max(cost1.qubits_max, cost1.qubits_out - cost2.qubits_in + cost2.qubits_max)

    return LogicalCosts(
        label=label,
        qubits_in=cost1.qubits_in,
        qubits_out=qubits_out,
        qubits_max=qubits_max,
        gates=cost1.gates + cost2.gates,
        depth=cost1.depth + cost2.depth,
        dw=dw,
        toffoli_count=cost1.toffoli_count + cost2.toffoli_count,
        t_count=cost1.t_count + cost2.t_count,
        t_depth=cost1.t_depth + cost2.t_depth,
    )


def compose_parallel(cost1, cost2, label="_"):
    # Pad wires from shallower circuit with identity gates
    dw = cost1.dw + cost2.dw
    if cost1.depth >= cost2.depth:
        dw += (cost1.depth - cost2.depth) * cost2.qubits_out
    else:
        dw += (cost2.depth - cost1.depth) * cost1.qubits_out

    return LogicalCosts(
        label=label,
        qubits_in=cost1.qubits_in + cost2.qubits_in,
        qubits_out=cost1.qubits_out + cost2.qubits_out,
        qubits_max=cost1.qubits_max + cost2.qubits_max,
        gates=cost1.gates + cost2.gates,
        depth=max(cost1.depth, cost2.depth),
        dw=dw,
        toffoli_count=cost1.toffoli_count + cost2.toffoli_count,
        t_count=cost1.t_count + cost2.t_count,
        t_depth=max(cost1.t_depth, cost2.t_depth),
    )


def classical_popcount_costf(n, k, metric):
    """
    Classical gate count for popcount.

    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    if metric == "naive_classical":
        cc = ClassicalCosts(label="popcount", gates=1, depth=1)
        return cc

    ell = mp.ceil(mp.log(n, 2))
    gates = n + (n - ell - 1)*5 + ell # 5 = gates per full adder
    depth = 2 * ell

    cc = ClassicalCosts(label="popcount", gates=gates, depth=depth)
    return cc


def adder_costf(i, ci=False):
    """
    Logical cost of i bit adder (Cuccaro et al). With Carry Input if ci=True

    """
    adder_cnots = 6 if i == 1 else (5 * i + 1 if ci else 5 * i - 3)
    adder_depth = 7 if i == 1 else (2 * i + 6 if ci else 2 * i + 4)
    adder_nots = 0 if i == 1 else (2 * i - 2 if ci else 2 * i - 4)
    adder_tofs = 2 * i - 1
    adder_qubits_in = 2 * i + 1
    adder_qubits_out = 2 * i + 2
    adder_qubits_max = 2 * i + 2
    adder_t_depth = adder_tofs * MagicConstants.t_depth_div_toffoli
    adder_t_count = adder_tofs * MagicConstants.t_div_toffoli
    adder_gates = adder_cnots + adder_nots + adder_tofs * MagicConstants.gates_div_toffoli

    return LogicalCosts(
        label=str(i) + "-bit adder",
        qubits_in=adder_qubits_in,
        qubits_out=adder_qubits_out,
        qubits_max=adder_qubits_max,
        gates=adder_gates,
        depth=adder_depth,
        dw=adder_qubits_in * adder_depth,
        toffoli_count=adder_tofs,
        t_count=adder_t_count,
        t_depth=adder_t_depth,
    )


def hamming_wt_costf(n):
    """
    Logical cost of mapping |v>|0> to |v>|H(v)>.

    ..  note :: The adder tree uses in-place addition, so some of the bits of |v> overlap |H(v)> and
    there are ancilla as well.

    :param n: number of bits in v

    """
    b = int(mp.floor(log2(n)))
    qc = null_costf(qubits_in=n, qubits_out=n)
    if bin(n + 1).count("1") == 1:
        # When n = 2**(b+1) - 1 the adder tree is "packed". We can use every input bit including
        # carry inputs.
        for i in range(1, b + 1):
            L = compose_k_parallel(adder_costf(i, ci=True), 2 ** (b - i))
            qc = compose_sequential(qc, L)
    else:
        # Decompose into packed adder trees joined by adders.
        # Use one adder tree on (2**b - 1) bits and one on max(1, n - 2**b) bits.
        # Reserve one bit for carry input of adder (unless n = 2**b).
        carry_in = n != 2 ** b
        qc = compose_sequential(
            qc, compose_parallel(hamming_wt_costf(2 ** b - 1), hamming_wt_costf(max(1, n - 2 ** b)))
        )
        qc = compose_sequential(qc, adder_costf(b, ci=carry_in))

    qc = compose_parallel(qc, null_costf(), label=str(n) + "-bit hamming weight")
    return qc


def carry_costf(m):
    """
    Logical cost of mapping |x> to (-1)^{(x+c)_m}|x> where (x+c)_m is the m-th bit (zero indexed) of
    x+c for an arbitrary m bit constant c.

    ..  note :: numbers here are adapted from Fig 3 of https://arxiv.org/pdf/1611.07995.pdf
                m is equivalent to ell in the LaTeX
    """
    if m < 2:
        raise NotImplementedError("Case m==1 not implemented.")

    carry_cnots = 2 * m
    carry_depth = 8 * m - 8
    carry_nots = 2 * (m - 1)
    carry_tofs = 4 * (m - 2) + 2
    carry_qubits_in = 2 * m
    carry_qubits_out = 2 * m
    carry_qubits_max = 2 * m
    carry_dw = carry_qubits_max * carry_depth
    carry_t_depth = carry_tofs * MagicConstants.t_depth_div_toffoli
    carry_t_count = carry_tofs * MagicConstants.t_div_toffoli
    carry_gates = carry_cnots + carry_nots + carry_tofs * MagicConstants.gates_div_toffoli

    return LogicalCosts(
        label="carry",
        qubits_in=carry_qubits_in,
        qubits_out=carry_qubits_out,
        qubits_max=carry_qubits_max,
        gates=carry_gates,
        depth=carry_depth,
        dw=carry_dw,
        toffoli_count=carry_tofs,
        t_count=carry_t_count,
        t_depth=carry_t_depth,
    )


def popcount_costf(L, n, k):
    """
    Logical cost of mapping |i> to (-1)^{popcount(u,v_i)}|i> for fixed u.

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    assert 0 <= k and k <= n

    index_wires = int(mp.ceil(log2(L)))

    # Initialize space for |v_i>
    qc = null_costf(qubits_in=index_wires, qubits_out=n + index_wires)

    # Query table index i
    # NOTE: We're skipping a qRAM call here.
    qc = delay(qc, 1)

    # XOR in the fixed sketch "u"
    # NOTE: We're skipping ~ n NOT gates for mapping |v> to |u^v>
    qc = delay(qc, 1)

    # Use tree of adders compute hamming weight
    #     |i>|u^v_i>|0>     ->    |i>|u^v_i>|wt(u^v_i)>
    hamming_wt = hamming_wt_costf(n)
    qc = compose_sequential(
        qc, null_costf(qubits_in=qc.qubits_out, qubits_out=index_wires + hamming_wt.qubits_in)
    )
    qc = compose_sequential(qc, hamming_wt)

    # Compute the high bit of (2^ceil(log(n)) - k) + hamming_wt
    #     |i>|v_i>|wt(u^v_i)>   ->     (-1)^popcnt(u,v_i) |i>|u^v_i>|wt(u^v_i)>
    qc = compose_sequential(qc, carry_costf(int(mp.ceil(log2(n)))))

    # Uncompute hamming weight.
    qc = compose_sequential(qc, reverse(hamming_wt))

    # Uncompute XOR
    # NOTE: We're skipping ~ n NOT gates for mapping |u^v> to |v>
    qc = delay(qc, 1)

    # Uncompute table entry
    # NOTE: We're skipping a qRAM call here.
    qc = delay(qc, 1)

    # Discard ancilla
    # (-1)^popcnt(u,v_i) |i>|0>|0>   ->    (-1)^popcnt(u,v_i) |i>

    qc = compose_sequential(qc, null_costf(qubits_in=qc.qubits_out, qubits_out=index_wires))

    qc = compose_parallel(qc, null_costf(), label="popcount" + str((n, k)))

    return qc


def n_toffoli_costf(n, have_ancilla=False):
    """
    Logical cost of toffoli with n-1 controls.

    ..  note :: Table I of Maslov arXiv:1508.03273v2 (Source = "Ours", Optimization goal = "T/CNOT")

    """

    assert n >= 3

    if n >= 5 and not have_ancilla:
        # Use Barenco et al (1995) Lemma 7.3 split into two smaller Toffoli gates.
        n1 = int(mp.ceil((n - 1) / 2.0)) + 1
        n2 = n - n1 + 1
        return compose_sequential(
            compose_parallel(
                null_costf(qubits_in=n - n1, qubits_out=n - n1), n_toffoli_costf(n1, True)
            ),
            compose_parallel(
                null_costf(qubits_in=n - n2, qubits_out=n - n2), n_toffoli_costf(n2, True)
            ),
        )

    if n == 3:  # Normal toffoli gate
        n_tof_t_count = MagicConstants.AMMR12_tof_t_count
        n_tof_t_depth = MagicConstants.AMMR12_tof_t_depth
        n_tof_gates = MagicConstants.AMMR12_tof_gates
        n_tof_depth = MagicConstants.AMMR12_tof_depth
        n_tof_dw = n_tof_depth * (n + 1)
    elif n == 4:
        """
        Note: the cost can be smaller if using "clean" ancillas
        (see first "Ours" in Table 1 of Maslov's paper)
        """
        n_tof_t_count = 16
        n_tof_t_depth = 16
        n_tof_gates = 36
        n_tof_depth = 36  # Maslov Eq. (5), Figure 3 (dashed), Eq. (3) (dashed).
        n_tof_dw = n_tof_depth * (n + 1)
    elif n >= 5:
        n_tof_t_count = 8 * n - 16
        n_tof_t_depth = 8 * n - 16
        n_tof_gates = (8 * n - 16) + (8 * n - 20) + (4 * n - 10)
        n_tof_depth = (8 * n - 16) + (8 * n - 20) + (4 * n - 10)
        n_tof_dw = n_tof_depth * (n + 1)

    n_tof_qubits_max = n if have_ancilla else n + 1

    return LogicalCosts(
        label=str(n) + "-toffoli",
        qubits_in=n,
        qubits_out=n,
        qubits_max=n_tof_qubits_max,
        gates=n_tof_gates,
        depth=n_tof_depth,
        dw=n_tof_dw,
        toffoli_count=0,
        t_count=n_tof_t_count,
        t_depth=n_tof_t_depth,
    )


def diffusion_costf(L):
    """
    Logical cost of the diffusion operator D R_0 D^-1

    where D samples the uniform distribution on {1,...,L} R_0 is the unitary I - 2|0><0|

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k

    """
    index_wires = int(mp.ceil(log2(L)))

    H = LogicalCosts(
        label="H",
        qubits_in=1,
        qubits_out=1,
        qubits_max=1,
        gates=1,
        depth=1,
        dw=1,
        toffoli_count=0,
        t_count=0,
        t_depth=0,
    )
    Hn = compose_k_parallel(H, index_wires)

    anc = null_costf(qubits_in=index_wires, qubits_out=index_wires + 1)

    qc = compose_sequential(Hn, anc)
    qc = compose_sequential(qc, n_toffoli_costf(index_wires + 1))
    qc = compose_sequential(qc, reverse(anc))
    qc = compose_sequential(qc, Hn)

    qc = compose_parallel(qc, null_costf(), label="diffusion")
    return qc


def popcount_grover_iteration_costf(L, n, k, metric):
    """
    Logical cost of G(popcount) = (D R_0 D^-1) R_popcount.

    where D samples the uniform distribution on {1,...,L} (D R_0 D^-1) is the diffusion operator.
    R_popcount maps |i> to (-1)^{popcount(u,v_i)}|i> for some fixed u

    :param L: length of the list, i.e. |L|
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on <= k

    """
    if metric == "naive_quantum":
        return LogicalCosts(
            label="oracle",
            qubits_in=1,
            qubits_out=1,
            qubits_max=1,
            depth=1,
            gates=1,
            dw=1,
            toffoli_count=1,
            t_count=1,
            t_depth=1,
        )

    popcount_cost = popcount_costf(L, n, k)
    diffusion_cost = diffusion_costf(L)

    return compose_sequential(diffusion_cost, popcount_cost, label="oracle")


def popcounts_dominate_cost(positive_rate, d, n, metric):
    ip_div_pc = (MagicConstants.word_size ** 2) * d / float(n)
    if metric in ClassicalMetrics:
        return 1.0 / positive_rate > ip_div_pc
    else:
        return 1.0 / positive_rate > ip_div_pc ** 2


def raw_cost(cost, metric):
    if metric == "g":
        result = cost.gates
    elif metric == "dw":
        result = cost.dw
    elif metric == "ge19":
        phys = estimate_abstract_to_physical(
            cost.toffoli_count,
            cost.qubits_max,
            cost.depth,
            prefers_parallel=False,
            prefers_serial=True,
        )
        result = cost.dw * phys[0] ** 2
    elif metric == "t_count":
        result = cost.t_count
    elif metric == "classical":
        result = cost.gates
    elif metric == "naive_quantum":
        return cost.gates
    elif metric == "naive_classical":
        return cost.gates
    elif metric == "hardware_time":
        return cost.gates
    else:
        raise ValueError("Unknown metric '%s'" % metric)
    return result


AllPairsResult = namedtuple(
    "AllPairsResult", ("d", "n", "k", "log_cost", "pf_inv", "eta", "metric", "detailed_costs")
)


def all_pairs(d, n=None, k=None, optimize=True, metric="dw", allow_suboptimal=False,list_size = None,kappangle=mp.pi/3):
    """
    Nearest Neighbor Search via a quadratic search over all pairs.

    :param d: search in S^{d-1}
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k
    :param optimize: optimize `n`
    :param metric: target metric
    :param allow_suboptimal: when ``optimize=True``, return the best possible set of parameters given what is precomputed
    :param list_size: 
    :param kappa:ngle the target overlap (two vectors are "reducing" if the angle between them is less than kappa)

    """
    if n is None:
        n = 1
        while n < d:
            n = 2 * n

    k = k if k else int(MagicConstants.k_div_n * (n - 1))

    pr = load_probabilities(d, n - 1, k,kappangle)

    def cost(pr):
        if not list_size:
            list_size = 2 / ((1 - pr.eta) * C(pr.d, kappangle))

        if metric in ClassicalMetrics:
            look_cost = classical_popcount_costf(pr.n, pr.k, metric)
            looks = (list_size ** 2 - list_size) / 2.0
            search_one_cost = ClassicalCosts(
                label="search", gates=look_cost.gates * looks, depth=look_cost.depth * looks
            )
        else:
            look_cost = popcount_grover_iteration_costf(list_size, pr.n, pr.k, metric)
            looks_factor = 11.0 / 15
            looks = int(mp.ceil(looks_factor * list_size ** (3 / 2.0)))
            search_one_cost = compose_k_sequential(look_cost, looks)

        full_cost = raw_cost(search_one_cost, metric)
        return full_cost, look_cost

    positive_rate = pf(pr.d, pr.n, pr.k)
    while optimize and not popcounts_dominate_cost(positive_rate, pr.d, pr.n, metric):
        try:
            pr = load_probabilities(
                pr.d, 2 * (pr.n + 1) - 1, int(MagicConstants.k_div_n * (2 * (pr.n + 1) - 1))
            )
        except PrecomputationRequired as e:
            if allow_suboptimal:
                break
            else:
                raise e
        positive_rate = pf(pr.d, pr.n, pr.k)

    fc, dc = cost(pr)

    return AllPairsResult(
        d=pr.d,
        n=pr.n,
        k=pr.k,
        log_cost=float(log2(fc)),
        pf_inv=int(1 / positive_rate),
        eta=pr.eta,
        metric=metric,
        detailed_costs=dc,
    )



## MOSTLY IGNORING
RandomBucketsResult = namedtuple(
    "RandomBucketsResult",
    ("d", "n", "k", "theta", "log_cost", "pf_inv", "eta", "metric", "detailed_costs"),
)


def random_buckets(
    d, n=None, k=None, theta1=None, optimize=True, metric="dw", allow_suboptimal=False
):
    """
    Nearest Neighbor Search using random buckets as in BGJ1.

    :param d: search in S^{d-1}
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k
    :param theta1: bucket angle
    :param optimize: optimize `n`
    :param metric: target metric
    :param allow_suboptimal: when ``optimize=True``, return the best possible set of parameters
        given what is precomputed

    """
    if n is None:
        n = 1
        while n < d:
            n = 2 * n

    k = k if k else int(MagicConstants.k_div_n * (n - 1))
    theta = theta1 if theta1 else 1.2860
    pr = load_probabilities(d, n - 1, k,compute=True)
    ip_cost = MagicConstants.word_size ** 2 * d

    def cost(pr, T1):
        eta = 1 - ngr_pf(pr.d, pr.n, pr.k, beta=T1) / ngr(pr.d, beta=T1)
        N = 2 / ((1 - eta) * C(pr.d, mp.pi / 3))
        W0 = W(pr.d, T1, T1, mp.pi / 3)
        m = log2(d)
        def cost_per_code(code_size):
            buckets = code_size
            num_codes = int(1.0 / (W0*code_size))
            bucket_size = N * C(pr.d, T1)


            if metric in ClassicalMetrics:
                look_cost = classical_popcount_costf(pr.n, pr.k, metric)
                looks_per_bucket = (bucket_size ** 2 - bucket_size) / 2.0
                search_one_cost = ClassicalCosts(
                    label="search",
                    gates=look_cost.gates * looks_per_bucket,
                    depth=look_cost.depth * looks_per_bucket,
                )
            else:
                look_cost = popcount_grover_iteration_costf(bucket_size, pr.n, pr.k, metric)
                looks_factor = (2 * W0) / (5 * C(pr.d, T1)) + 1.0 / 3
                looks_per_bucket = int(looks_factor * bucket_size ** (3 / 2.0))
                search_one_cost = compose_k_sequential(look_cost, looks_per_bucket)

            memory = max(bucket_size*buckets, N*(m*32))*(m*32)
            mem_cost =  pow(memory,1.5)* MagicConstants.BIT_OPS_PER_SORT_BIT


            fill_bucket_cost = N * ip_cost
            search_bucket_cost = raw_cost(search_one_cost, metric)
            single_code_cost = buckets * (fill_bucket_cost + search_bucket_cost) + mem_cost
            return single_code_cost*num_codes, look_cost, eta
        best_code_size = local_min(lambda C: cost_per_code(C)[0], low=1.0,high=1.0/W0)
        return cost_per_code(best_code_size)


    if optimize:
        theta = local_min(lambda T: cost(pr, T)[0], low=mp.pi / 6, high=mp.pi / 2)
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
        while not popcounts_dominate_cost(positive_rate, pr.d, pr.n, metric):
            try:
                n = 2 * (pr.n + 1) - 1
                k = int(MagicConstants.k_div_n * n)
                pr = load_probabilities(pr.d, n, k,compute=True)
            except PrecomputationRequired as e:
                if allow_suboptimal:
                    break
                else:
                    raise e
            theta = local_min(lambda T: cost(pr, T)[0], low=mp.pi / 6, high=mp.pi / 2)
            positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)
    else:
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta)

    fc, dc, eta = cost(pr, theta)

    return RandomBucketsResult(
        d=pr.d,
        n=pr.n,
        k=pr.k,
        theta=float(theta),
        log_cost=float(log2(fc)),
        pf_inv=int(1 / positive_rate),
        eta=eta,
        metric=metric,
        detailed_costs=dc,
    )


ListDecodingResult = namedtuple(
    "ListDecodingResult",
    ("d", "n", "k", "theta", "log_cost","code_size",  "pf_inv", "eta", "metric", "m","num_codes","filter_cost","memory_cost","search_cost", "list_memory", "bucket_memory"),
)




"""
Nearest Neighbor Search via a decodable buckets as in BDGL16.

:param d: search in S^{d-1}
:param n: number of entries in popcount filter
:param k: we accept if two vectors agree on ≤ k
:param theta: array of filter creation angle
:param given_code_size: array of code sizes to use
:param optimize: optimize `n` and `theta`
:param metric: target metric
:param allow_suboptimal: when ``optimize=True``, return the best possible set of parameters
    given what is precomputed
:param recursion_depth: number of recursive list decoding searches allowed, including this one

This function is a wrapper function. If optimize is false, it calls 
`list_decoding_internal` directly.
Otherwise, it uses the results from an optimized call -- which
are imprecise, to save computation -- as inputs
to a more precise call to `list_decoding_internal`
"""
def list_decoding(
    d, 
    n=None, 
    k=None, 
    theta=None, 
    given_code_size=None, 
    optimize=True, 
    metric="classical", 
    allow_suboptimal=False, 
    recursion_depth = 1
): 

    results =list_decoding_internal(
        d=d,
        n_in=n,
        k_in=k,
        theta_in=theta,
        given_code_size=given_code_size,
        fast=optimize,
        metric=metric,
        allow_suboptimal=allow_suboptimal,
        list_size=None,
        kappangle=mp.pi/3,
        recursion_depth=recursion_depth,
        exact=True)
    if not optimize:
        return results
    # Unwrap the optimal parameters for each level of recursion
    thetas = [results[3]]
    ns = [results[1]]
    ks = [results[2]]
    code_sizes = [2**results[5]]
    while type(results[13]) == ListDecodingResult:
        results = results[13]
        thetas = thetas + [results[3]]
        ns = ns + [results[1]]
        ks = ks + [results[2]]
        code_sizes = code_sizes + [2**results[5]]
    return list_decoding_internal(d,ns,ks,thetas,code_sizes,False,metric,allow_suboptimal,None,mp.pi/3,recursion_depth,True)



def list_decoding_internal(
    d, 
    n_in=None, 
    k_in=None, 
    theta_in=None, 
    given_code_size=None, 
    fast=False, 
    metric="classical", 
    allow_suboptimal=False, 
    list_size = None, 
    kappangle = mp.pi / 3, 
    recursion_depth = 1,
    exact = True,
):
    """
    Nearest Neighbor Search via a decodable buckets as in BDGL16.

    :param d: search in S^{d-1}
    :param n: number of entries in popcount filter
    :param k: we accept if two vectors agree on ≤ k
    :param theta_in: array of filter creation angle
    :param given_code_size: array of code sizes to use
    :param fast: takes approximate answers for things like false positive probabilities
    :param metric: target metric
    :param allow_suboptimal: when ``optimize=True``, return the best possible set of parameters
        given what is precomputed
    :param recursion_depth: number of recursive list decoding searches allowed, including this one
    :param exact: internal parameter mostly, governs how careful it is when optimizing parameters
    """

    if n_in:
        n = n_in
    else:
        n = 1
        while n < d:
            n = 2 * n
        n = [n-1]

    k = k_in if k_in else [int(MagicConstants.k_div_n * (n[0]))]
    theta = theta_in if theta_in else [mp.pi / 3]
    pr = load_probabilities(d, n[0], k[0], compute = True)
    def cost(pr, T1,local_list_size, _exact):
        sub_results = []

        C0 = C(d,kappangle)
        W0 = Wmatched(d, T1[0], kappangle,_exact) #
        CT1 = C(d,T1[0])

        # This computation is *so* slow, especially for large beta
        # that we simply ignore it for optimizing
        
        if _exact and not fast:
            eta = 1 - ngr_pf(pr.d, pr.n, pr.k, beta=T1[0], integrate=_exact) / ngr(pr.d, beta=T1[0],kappangle=kappangle,integrate=_exact)# some sort of probabilistic constant?
            if eta < 0:
                print("Eta problem: ",eta)
                print(ngr(pr.d, beta=T1[0],kappangle=kappangle,integrate=_exact))
                print(ngr_pf(pr.d, pr.n, pr.k, beta=T1[0], integrate=_exact))
                eta = 0
        else:
            eta = 0.75
        if not local_list_size:
            local_list_size = 2 / ((1 - eta) * C0) # list size
        


        list_memory = local_list_size*(d*MagicConstants.word_size + n[0]) # using 32 bits for each element of each vector
                                   # plus n bits to keep pre-computed popcounts with the vector
        # Guess at a code size to match the size of the list
        #  total bucket elements = (code_size * C(d, T1)) * N
        #  each element is one vector (d*32 bits) and one codeword is a hash, let's say 2*lg(C)+20 
        #  thus we want N*(d*32) = code_size * C(d,T1) * N * C(d,T2) * (2*lg(C) + 20)
        #  We approximate as follows:
        matched_code_size = (d*MagicConstants.word_size)/(CT1*(2*log2((d*MagicConstants.word_size)/CT1)+MagicConstants.COLLISION_BUFFER))
        # Re-calculate the actual bucket sizes
        one_bucket_size = max(local_list_size * CT1,1)
        
        # Step one: Get the cost to search one bucket
        if metric in ClassicalMetrics:
            # Try if a recursion works
            # This will be the new criteria for when vectors are reducing
            alpha_sq = mp.cos(T1[0])**2
            new_kappangle = mp.acos((mp.cos(kappangle) - alpha_sq)/(1-alpha_sq)) 
            # The kappangle check ensures we have not jumped out of a feasible range 
            if recursion_depth > 1 and new_kappangle > 0 and new_kappangle < mp.pi / 2:
                # First: we need to compute the vector proj_c(v) for each codeword c
                projection_cost = one_bucket_size*d* MagicConstants.word_size ** 2
                # Then we search the projections for reducing pairs
                # If we were given n, k, theta, code_size, this uses the remaining values
                recursion_cost_full = list_decoding_internal(
                    d=d-1, 
                    n_in=n_in[1:] if n_in else None, 
                    k_in=k_in[1:] if k_in else None,
                    theta_in=T1[1:],
                    given_code_size = given_code_size[1:] if given_code_size else given_code_size,
                    fast = fast, 
                    metric = metric, 
                    allow_suboptimal = allow_suboptimal, 
                    list_size = one_bucket_size,
                    kappangle=new_kappangle, 
                    recursion_depth = recursion_depth - 1,
                    exact = _exact)
                recursion_cost = 2.**recursion_cost_full[4] + projection_cost
                # This takes the search cost to be the recursive cost
                # If this is suboptimal (i.e., we're better off with a naive search)
                # this will ignore that fact!
                sub_results = recursion_cost_full
                search_cost = recursion_cost
                # Adjustment based on false positive rate, which we assume is at most 0.5
                if fast:
                    search_cost *= 2
                else:
                    search_cost /= p_recursion_hit(d, T1[0], kappangle)
                sub_memory = 2**max(recursion_cost_full[14],recursion_cost_full[15])
            else:
                # Finished recursing; just take the exhaustive search cost
                look_cost = classical_popcount_costf(pr.n, pr.k, metric)
                looks_per_bucket = max(1,one_bucket_size*(one_bucket_size-1)/2)
                search_one_cost = ClassicalCosts(
                    label="search",
                    gates=look_cost.gates * looks_per_bucket,
                    depth=look_cost.depth * looks_per_bucket,
                )
                search_cost = raw_cost(search_one_cost, metric)
                sub_results = float(log2(search_cost))
                sub_memory = 0
        else:
            raise ValueError("Quantum cost metrics no longer supported")

        # Step 2: Find the best code size
        # Notice that the search cost per bucket is independent of the code size; that's
        # why we already computed it.
        # Here we will define a separate function to compute the cost for each code, then
        # optimize for that
        # First, compute some values that do not depend on code size
        necessary_solutions = C0*local_list_size*(local_list_size - 1)/2 # expected number of reducing pairs
        def cost_per_code(pr,T1,local_list_size,local_code_size, _code_exact):
            # Compute memory for all the buckets
            # all_buckets_size = local_code_size * one_bucket_size
            # Each element of each bucket contains one vector and one hash of a codeword
            bucket_memory = one_bucket_size * (d*MagicConstants.word_size + n[0]) #+ 2*log2(local_code_size) + MagicConstants.COLLISION_BUFFER)
            bucket_memory = local_code_size * max(bucket_memory, sub_memory)
            filters = local_code_size # previously: 1.0 / W0
            # We expect the number of results per code to be 
            # local_code_size * (bucket size choose 2) * prob[two vectors reduce | two vectors in same bucket]
            # The probability is equal to W0 * C(d,kappangle) / C(d,T1[0])**2
            # The bucket size equals local_list_size * C(d,T1[0])
            # So the C(d,T1[0]) factors cancel out, except for a factor of 2
            # The reason we cancel out instead of using the actual bucket size
            # is that the bucket size is taken as at least 1 for memory management,
            # and this would falsely assume bucket sizes less than 1 on average can 
            # produce as many reducing vectors as a bucket of size 1
            solutions_per_code = local_code_size * local_list_size**2 * W0 * C0 / 2
            # We need a certain number of codes
            # We use binomial Chernoff bounds with the slightly erroneous assumption
            # that pairs of vectors are independently likely to be a reducing pair
            b = 3*mp.log(MagicConstants.PROB_MIN) - 2*necessary_solutions
            necessary_exp_val  = (-b + mp.sqrt(b**2 - 4*necessary_solutions**2))/2
            num_codes = max(1.0, necessary_exp_val/solutions_per_code)


            # Memory cost
            if metric == "hardware_time":
                # 2-D sort cost 
                # print("---Depth ",recursion_depth,"code size",log2(local_code_size),"---")
                # print(log2(list_memory),log2(bucket_memory))
                mem_cost = pow(bucket_memory,1+MagicConstants.SORT_EXPONENT_DELTA)
                mem_cost += pow(list_memory,1+MagicConstants.SORT_EXPONENT_DELTA)
                mem_cost *= MagicConstants.BIT_OPS_PER_SORT_BIT
                # Basic sort cost, for small sizes
                mem_cost += (bucket_memory*log2(bucket_memory)+list_memory*log2(list_memory))*MagicConstants.SORT_CONSTANT
            else:
                mem_cost = 0

            # We optimize for the number of random products in the random product code
            # "Optimize" means decrease m until it is close to the memory cost
            
            def cost_to_filter(_m):
                # Cost of an inner product with the code words
                ip_cost = d/_m * MagicConstants.word_size ** 2
                Z = filters**(1/_m) # number of vectors per subcode

                # we assume a cost of one word addition (five gates per bit)
                # + dealing with a pointer into asubcode per iteration node.
                COST_TREE_ITER    = 5 * MagicConstants.word_size + log2(Z)
                # we assume a cost of one word operation for the sorting + dealing with a pointer
                COST_COMPARE_SWAP = MagicConstants.word_size + log2(Z)

                # cost of inner products and cost of sorting the lists
                preprocess_cost = _m * Z * ip_cost  +  _m * Z * log2(Z) * COST_COMPARE_SWAP

                # We assume the enumeration procedure from the "Report on the Security of LWE: Improved Dual
                # Lattice Attack" https://doi.org/10.5281/zenodo.6412487 such that number of enumeration
                # nodes is a constant multiple of the number of solutions.

                insert_cost = preprocess_cost + filters * CT1 * COST_TREE_ITER
                return insert_cost

            # Binary search for lowest possible m
            # We want the filter cost to be at most 
            # 1/4 the memory cost, with "1/4"
            # an arbitrary constant just to keep costs low
            m=int(log2(d))
            if metric == "hardware_time":
                high_m = int(log2(d))
                low_m = 1
                while high_m - low_m > 1:
                    m = int((high_m + low_m)/2)
                    if local_list_size*cost_to_filter(m) < mem_cost/4:
                        high_m = m
                    else:
                        low_m = m
                if local_list_size*cost_to_filter(m) >= mem_cost/4:
                    m = m+1
            filter_cost = cost_to_filter(m)

            # Quick sanity checks
            if local_list_size < 0:
                raise ValueError("LIST SIZE ERROR")
            elif filter_cost < 0:
                raise ValueError("FILTER COST ERROR")

            # Return basic parameters
            return [num_codes,local_list_size*filter_cost,mem_cost], m, bucket_memory

        # This does a binary search for the code
        # if no code size is provided
        if given_code_size is None:
            low_code = 1
            high_code = 1/W0
            if _exact:
                code_bound = 1e-3
            else:
                code_bound = 1
            while abs(log2(low_code) - log2(high_code)) > code_bound and abs(high_code - low_code) > 1:
                mid_code = int(mp.sqrt(low_code*high_code)+0.5)
                # When we're in the initial steps of the search,
                # it is pointless to use expensive, high-precision
                # computations, so code_exact suppresses this
                code_exact = (abs(log2(low_code) - log2(high_code)) < 1)
                costs, _, bucket_memory = cost_per_code(pr,T1, local_list_size,mid_code, code_exact and _exact)


                if costs[2] > max(costs[1],search_cost*mid_code):  
                    if bucket_memory > list_memory:
                        # memory costs dominate
                        high_code = mid_code
                    else:
                        # memory costs dominate but just from list memory
                        # thus the cost decreases with fewer codes
                        low_code = mid_code
                elif costs[1] > search_cost*mid_code:
                    # query cost is the greatest
                    # this should decrease with code size
                    low_code = mid_code
                else:
                    # search cost is greatest
                    # this grows with the code size 
                    # if the number of codes is 1
                    if costs[0] == 1:
                        high_code = mid_code
                    else:
                        # otherwise search cost is independent of code size
                        # but this case has search cost greatest
                        # so there's nothing we can do
                        high_code = mid_code
                        low_code = mid_code
            best_code_size = low_code
        else:
            best_code_size = given_code_size[0]

        # Sanity checks
        if best_code_size < 0:
            raise ValueError("CODE SIZE ERROR")
        if search_cost < 0:
            raise ValueError("SEARCH COST ERROR")

        # Final call to get the actual cost
        (costs,m, bucket_memory) = cost_per_code(pr, T1, local_list_size,best_code_size,_exact)
        # These are the costs and parameters for one value of the filter angle T1 
        return costs+[best_code_size*search_cost], m, eta, best_code_size, sub_results, list_memory, bucket_memory

    # Step 3: optimize theta (if necessary) and compute the cost in that case
    if not theta_in:
        # We won't rely on the built-in optimizer because we know more about the structure
        # It seems to have roughly one global minimum in this region
        # So we assume it increases and then decreases
        low_theta = mp.acos(mp.sqrt(mp.cos(kappangle)))
        high_theta = mp.pi / 2
        past_local_min = False
        
        theta = (low_theta+high_theta)/2
        low_val = mp.inf
        high_val = mp.inf
        cost_ =cost(pr, [theta],list_size, False)[0]
        mid_val = cost_[0]*(cost_[1]+cost_[2]+cost_[3])
        # Configure progress bar:
        if DisplayConfig.display:
            pbar_length = log2(high_theta - low_theta)
            pbar = tqdm(total=int(1.5*(pbar_length - log2(1e-5))), leave=False, desc = str("Theta ")+str(recursion_depth))
        theta_exact = False
        while abs(high_theta - low_theta) > 1e-5:
            # Idea of this search: we maintain that the minimum is between high_theta and low_theta
            # Assuming the function decreases down to the minimum, then increases
            # we can preserve this property if low_val > mid_val < high_val
            # Thus, we split one of the two intervals with a new angle new_theta
            # This gives us four values (low_val, mid_val, new_val, high_val) 
            # (or (low_val, new_val, mid_val, high_val) if we chose new_theta < theta)
            # Either mid_val or new_val is the new minimum, so we take the three
            # values around the new minimum for the next iteration

            # Roughly, this tells us in which directions the function is increasing
            # or decreasing
            # We take as large of a cut as possible because:
            #    (a) it converges faster
            #    (b) the cost function is not perfectly increasing/decreasing
            #        at a small scale, so this lets us catch larger trends
            #        and not get as badly stuck in local minima

            # Here we check whether we should start being exact
            if not theta_exact and (abs(high_theta - low_theta) < 1e-1):
                theta_exact = True
                # Recompute mid_val so it can be appropriately compared
                cost_ =cost(pr, [theta],list_size, True)[0]
                mid_val = cost_[0]*(cost_[1]+cost_[2]+cost_[3])
            low_split = True # did we split the lower interval?
            new_theta = (low_theta + high_theta)/2
            # if new_theta is too close to theta, adjust
            if abs(new_theta - theta) < 1e-2*(high_theta - low_theta):
                # Guessing that the minimum is closer to the lower value
                if high_val < low_val:
                    new_theta = (2*high_theta + low_theta)/3
                else:
                    new_theta = (2*low_theta + high_theta)/3
            elif new_theta > theta:
                low_split = False
            cost_ =cost(pr, [new_theta],list_size, exact and theta_exact)[0]
            new_val = cost_[0]*(cost_[1]+cost_[2]+cost_[3])
            # Narrow down to the new interval based on where we split
            # and which value is the minimum so far
            if new_val < mid_val:
                if low_split:
                    high_theta = theta
                    theta=new_theta
                    high_val = mid_val
                    mid_val = new_val
                else:
                    low_theta = theta
                    theta=new_theta
                    low_val = mid_val
                    mid_val = new_val
            else:
                if low_split:
                    low_theta = new_theta
                    low_val = new_val
                else:
                    high_theta = new_theta
                    high_val = new_val
            if DisplayConfig.display:
                pbar.update(1)
        if DisplayConfig.display:
            pbar.close()

        
        theta = [theta]

    # If we're running this fast, we do not care about the false positive rate
    if fast:
        positive_rate = 1
    else:
        positive_rate = pf(pr.d, pr.n, pr.k, beta=theta[0],integrate = exact)
    if not n_in and not fast:
        # If we were not explicitly given popcount parameters,
        # we attempt to optimize them
        # Too many filters makes too much computation, but too few requires
        # too many re-computations
        # Modified from the original to not re-optimize theta
        # Rough idea: the cost barely changes with alternative pop-count figures
        # So we are not going to bother re-optimizing: it would be far too slow
        while not popcounts_dominate_cost(positive_rate, pr.d, pr.n, metric):
            pr = load_probabilities(
                pr.d, 2 * (pr.n + 1) - 1, int(MagicConstants.k_div_n * (2 * (pr.n + 1) - 1))
                , compute=True
            )
            positive_rate = pf(pr.d, pr.n, pr.k, beta=theta[0])

    # All optimization is done, we now call the result
    '''
        all_costs: an array of 4 elements:
            - number of codes
            - filter cost
            - memory cost
            - search cost
        m: optimal m for product code
        eta: fraction of reducing vectors found by popcount filters
        code_size: the code size
        sub_results: list_decoding results from any recursive calls
    '''
    all_costs, m, eta, code_size, sub_results, list_mem, bucket_mem = cost(pr, theta,list_size, exact)

    # full cost
    fc = all_costs[0]*(all_costs[1]+all_costs[2]+all_costs[3])

    return ListDecodingResult(
        d=pr.d,
        n=pr.n,
        k=pr.k,
        theta=float(theta[0]),
        log_cost=log2(fc),
        code_size = log2(code_size),
        pf_inv=int(1 / positive_rate),
        eta=float(eta),
        metric=metric,
        m=m,
        num_codes = float(log2(all_costs[0])),
        filter_cost = float(log2(all_costs[1])),
        memory_cost = float(log2(all_costs[2])),
        search_cost = sub_results,
        list_memory = float(log2(list_mem)),
        bucket_memory = float(log2(bucket_mem)),
    )


# Returns a list of headers for a csv file
# It includes extra columns based on the total depth of recursion
def list_decoding_title(recursion_depth):
    title = ["dimension", "metric","memory_cost","recursion_depth","total cost"]
    for i in range(recursion_depth):
        next_row = ["n","k","theta", "code_size","list size", "buckets size", "m","eta","pf_inv", "num_codes", "filter_cost","memory_cost", "search_cost"]
        title += [header + "_" + str(i) for header in next_row]
    return title

# Given a ListDecodingResult argument, returns a list of results
# in the expected order to be output to CSV
def list_decoding_as_list(ld):
    csv_row = [ld[0],ld[8],log2(MagicConstants.BIT_OPS_PER_SORT_BIT)]
    sub_row = [float(ld[4])]
    flag = True
    depth = 0
    while flag:
        depth += 1
        sub_row += [ld[1], ld[2], ld[3], float(ld[5]), ld[14], ld[15], ld[9],ld[7],ld[6], ld[10],ld[11], ld[12]]
        search_cost = ld[5]
        flag = (type(ld[13]) == ListDecodingResult) # is the search cost another listdecoding?
        if flag:
            search_cost += ld[13][4] # check the total cost of the subroutine
        else:
            search_cost += ld[13] # search cost is given directly
        sub_row += [float(search_cost)]
        ld = ld[13]
    return csv_row + [depth] + sub_row 

SieveSizeResult = namedtuple("SieveSizeResult", ("d", "log2_size", "metric", "detailed_costs"))


def sieve_size(d, metric=None):
    N = 2 / (C(d, mp.pi / 3))
    if metric == "vectors":
        log2_size = log2(N)
    elif metric == "bits":
        log2_size = log2(N) + log2(d)
    return SieveSizeResult(d=d, log2_size=log2_size, metric=metric, detailed_costs=(0,))

