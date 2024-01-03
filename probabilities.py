# -*- coding: utf-8 -*-
"""
Estimating relevant probabilities on the sphere and for popcount.

To run doctests, run: ``PYTHONPATH=`pwd` sage -t probabilities.py``

"""
from mpmath import mp, mpf
from tqdm import tqdm
from collections import namedtuple
from functools import partial
from memoize import memoize
from multiprocessing import Pool
from config import MultiProcessingConfig

Probabilities = namedtuple(
    "Probabilities", ("d", "n", "k", "gr", "ngr", "pf", "ngr_pf", "gr_pf", "rho", "eta", "beta", "prec")
)

class DisplayConfig:
    display = True


def multi_integrals(fs_and_bounds):
    fs = fs_and_bounds[0]
    bounds = fs_and_bounds[1]
    results = [[]]*len(fs)
    for i in range(len(fs)):
        result = [0]*len(bounds[i])
        for ii in range(len(bounds[i])):
            result[ii] = mp.quad(fs[i], bounds[i][ii],error=True)
        results[i] = result
    return results


def C(d, theta, integrate=False, prec=None):
    """
    The probability that some v from the sphere has angle at most θ with some fixed u.

    :param d: We consider spheres of dimension `d-1`
    :param theta: angle in radians
    :param: compute via explicit integration
    :param: precision to use
    :param approx: 

    EXAMPLE::

        sage: C(80, pi/3)
        mpf('1.0042233739846629e-6')

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        if integrate:
            r = (
                1
                / mp.sqrt(mp.pi)
                * mp.gamma(d / 2)
                / mp.gamma((d - 1) / 2)
                * mp.quad(lambda x: mp.sin(x) ** (d - 2), (0, theta), error=True)[0]
            )
            r_err = mp.quad(lambda x: mp.sin(x) ** (d - 2), (0, theta), error=True)
            # print("error: ",r_err[1]/r_err[0])
        else:
            r = mp.betainc((d - 1) / 2, 1 / 2.0, x2=mp.sin(theta) ** 2, regularized=True) / 2
        return r


def A(d, theta, prec=53):
    """
    The density of the event that some v from the sphere has angle θ with some fixed u.

    :param d: We consider spheres of dimension `d-1`
    :param theta: angle in radians

    :param: compute via explicit integration
    :param: precision to use

    EXAMPLES::

        sage: A(80, pi/3)
        mpf('4.7395659506025816e-5')

        sage: A(80, pi/3) * 2*pi/100000
        mpf('2.9779571143234787e-9')

        sage: C(80, pi/3+pi/100000) - C(80, pi/3-pi/100000)
        mpf('2.9779580567976835e-9')

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        r = 1 / mp.sqrt(mp.pi) * mp.gamma(d / 2) / mp.gamma((d - 1) / 2) * mp.sin(theta) ** (d - 2)
        return r


@memoize
def log2_sphere(d):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return (d / 2 * mp.log(mp.pi, 2) + 1) / mp.gamma(d / 2)


@memoize
def sphere(d):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return 2 ** (d / 2 * mp.log(mp.pi, 2) + 1) / mp.gamma(d / 2)


@memoize
def W(d, alpha, beta, theta, integrate=True, prec=None):
    assert alpha <= mp.pi / 2
    assert beta <= mp.pi / 2
    assert 0 >= (mp.cos(beta) - mp.cos(alpha) * mp.cos(theta)) * (mp.cos(beta) * mp.cos(theta) - mp.cos(alpha))

    if theta >= alpha + beta:
        return mp.mpf(0.0)

    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        alpha = mp.mpf(alpha)
        beta = mp.mpf(beta)
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        if integrate:
            c = mp.atan(mp.cos(alpha) / (mp.cos(beta) * mp.sin(theta)) - 1 / mp.tan(theta))

            def f_alpha(x):
                return mp.sin(x) ** (d - 2) * mp.betainc(
                    (d - 2) / 2,
                    1 / 2.0,
                    x2=mp.sin(mp.re(mp.acos(mp.tan(theta - c) / mp.tan(x)))) ** 2,
                    regularized=True,
                )

            def f_beta(x):
                return mp.sin(x) ** (d - 2) * mp.betainc(
                    (d - 2) / 2, 1 / 2.0, x2=mp.sin(mp.re(mp.acos(mp.tan(c) / mp.tan(x)))) ** 2, regularized=True
                )
            R_alpha = mp.quad(f_alpha, (theta - c, alpha), error=True)
            R_beta = mp.quad(f_beta, (c, beta), error=True)
            S_alpha = R_alpha[0]/2
            S_beta = R_beta[0]/2
            # print(R_alpha[1]/R_alpha[0], R_beta[1],R_beta[0])
            return (S_alpha + S_beta) * sphere(d - 1) / sphere(d)
        else:
            # Wedge volume formula from Lemma 2.2 of [BDGL16] Anja Becker, Léo Ducas, Nicolas Gama,
            # Thijs Laarhoven. "New directions in nearest neighbor searching with applications to
            # lattice sieving." SODA 2016. https://eprint.iacr.org/2015/1128
            g_sq = (mp.cos(alpha)**2 + mp.cos(beta)**2 -
            2*mp.cos(alpha)*mp.cos(beta)*mp.cos(theta))/mp.sin(theta)**2
            log2_A = mp.log(g_sq, 2) - 2*mp.log(1-g_sq, 2)
            r = (d-4) * mp.log(mp.sqrt(1-g_sq), 2) + log2_A - 2*mp.log(d-4, 2) + log2_sphere(d-2) - log2_sphere(d)
            return 2**r

# Same as W, but optimized for the case of alpha=beta
@memoize
def Wmatched(d, alpha, theta, integrate=True, prec=None):
    assert alpha <= mp.pi / 2
    assert alpha <= mp.pi / 2
    # assert 0 >= (mp.cos(alpha) - mp.cos(alpha) * mp.cos(theta)) * (mp.cos(alpha) * mp.cos(theta) - mp.cos(alpha))

    if theta >= 2*alpha:
        return mp.mpf(0.0)

    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        alpha = mp.mpf(alpha)
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        if integrate:
            c = mp.atan(mp.cos(alpha) / (mp.cos(alpha) * mp.sin(theta)) - 1 / mp.tan(theta))

            def f_alpha(x):
                return mp.sin(x) ** (d - 2) * mp.betainc(
                    (d - 2) / 2,
                    1 / 2.0,
                    x2=mp.sin(mp.re(mp.acos(mp.tan(theta - c) / mp.tan(x)))) ** 2,
                    regularized=True,
                )

            def f_beta(x):
                return mp.sin(x) ** (d - 2) * mp.betainc(
                    (d - 2) / 2, 1 / 2.0, x2=mp.sin(mp.re(mp.acos(mp.tan(c) / mp.tan(x)))) ** 2, regularized=True
                )

            R_alpha = mp.quad(f_alpha, (theta - c, alpha), error=True)
            R_beta = mp.quad(f_beta, (c, alpha), error=True)
            S_alpha = R_alpha[0]/2
            S_beta = R_beta[0]/2
            # print("Errors:", R_alpha[1]/R_alpha[0], R_beta[1]/R_beta[0])
            # S_alpha = 0
            # S_beta = 0
            # bound = 8
            # lb_a = theta - c
            # r_a = alpha - lb_a
            # lb_b = c
            # r_b = alpha - lb_b
            # for i in range(bound):
            #     S_alpha += mp.quad(f_alpha, [lb_a + i*r_a/bound,lb_a+(i+1)*r_a/bound],error=True)[0]
            #     S_beta += mp.quad(f_beta, [lb_b + i*r_b/bound,lb_b+(i+1)*r_b/bound],error=True)[0]
                
            return (S_alpha + S_beta) * sphere(d - 1) / sphere(d)
        else:
            # Wedge volume formula from Lemma 2.2 of [BDGL16] Anja Becker, Léo Ducas, Nicolas Gama,
            # Thijs Laarhoven. "New directions in nearest neighbor searching with applications to
            # lattice sieving." SODA 2016. https://eprint.iacr.org/2015/1128
            g_sq = (2*mp.cos(alpha)**2 *(1 -mp.cos(theta)))/mp.sin(theta)**2
            log2_A = mp.log(g_sq, 2) - 2*mp.log(1-g_sq, 2)
            r = (d-4) * mp.log(mp.sqrt(1-g_sq), 2) + log2_A - 2*mp.log(d-4, 2) + log2_sphere(d-2) - log2_sphere(d)
            return 2**r


# Probability that, given two vectors with inner product at least ck,
# a randomly chosen unit vector will be at angle at most alpha with
# both. The difference with the function `W` is that the `W` assumes
# the inner product is *exactly* ck
@memoize
def Wfull(d,alpha,ck,prec=53):
    def f(x,y):
        cx = mp.cos(x)
        cy = mp.cos(y)
        num = ck - cx*cy
        den = mp.sqrt((1-cx**2)*(1-cy**2))
        # We want the probability that they
        # are reducing, given that the inner angle is 
        # at least kappangle
        if num >= 0:
            if num >= den: # cannot be reducing
                return 0
            else:
                # Could be reducing 
                return C(d-1,mp.acos(num/den),prec)*A(d,x,prec)*A(d,y,prec)
        else:
            if num <= -den:
                # vectors are so close to the code they must be reducing
                return A(d,x,prec)*A(d,y,prec)
            else:
                return (0.5 + C(d-1,mp.acos(-num/den),prec))*A(d,x,prec)*A(d,y,prec)
    return mp.quad(f, [0,alpha],[0,alpha],error=True)





@memoize
def p_hit_kappa(d,alpha,ck,kappangle,prec=53):
    def f(x,y):
        cx = mp.cos(x)
        cy = mp.cos(y)
        num = ck - cx*cy
        den = mp.sqrt((1-cx**2)*(1-cy**2))
        # We want the probability that they
        # are reducing, given that the inner angle is 
        # at least kappangle
        if num >= 0:
            if num >= den: # cannot be reducing
                return 0
            else:
                # Could be reducing for all kappa angles less than 
                # num/den 
                angle = mp.acos(num/den)
                return (C(d-1,min(angle,kappangle),prec))*A(d,x,prec)*A(d,y,prec)
        else:
            # vectors are so close to the code they must be reducing
            return C(d-1,kappangle,prec)*A(d,x,prec)*A(d,y,prec)

    return mp.quad(f, [0,alpha],[0,alpha],error=True)

# Computes the "false negative" probability
# of the recursion; i.e., the probability
# that two vectors in a filter bucket of angle
# theta, which are at angle kappangle with each other,
# will not be detected by the recursive subroutine
@memoize
def p_recursion_hit(d, alpha, kappangle):
    alpha_sq = mp.cos(alpha)**2
    ck = mp.cos(kappangle)
    sub_kappa = mp.acos((ck - alpha_sq)/(1-alpha_sq)) 
    N, Nerr = p_hit_kappa(d, alpha, ck, sub_kappa)
    D, Derr = Wfull(d,alpha, ck)
    return N/D


@memoize
def binomial(n, i):
    # NOTE: hardcoding 53 here
    with mp.workprec(53):
        return mp.binomial(n, i)


@memoize
def P(n, k, theta, prec=None):
    """
    Probability that two vectors with angle θ pass a popcount filter

    :param n: number of popcount vectors
    :param k: number of popcount tests required to pass
    :param theta: angle in radians

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        theta = mp.mpf(theta)
        # binomial cdf for 0 <= successes <= k
        # r = 0
        # for i in range(k):
        #     r += binomial(n, i) * (theta/mp.pi)**i * (1-theta/mp.pi)**(n-i)
        # r = mp.betainc(n-k, k+1, x2=1-(theta/mp.pi), regularized=True)
        # NOTE: This routine uses obscene precision

        def _betainc(a, b, x2):
            return (
                x2 ** a
                * mp.hyp2f1(
                    a, 1 - b, a + 1, x2, maxprec=2 ** mp.ceil(2 * mp.log(n, 2)), maxterms=2 ** mp.ceil(mp.log(n, 2))
                )
                / a
                / mp.beta(a, b)
            )

        r = _betainc(n - k, k + 1, x2=1 - (theta / mp.pi))
        return r

def ngr_internal_function(d,beta,integrate,x):
    return Wmatched(d,beta,x,integrate)*A(d,x)

def pf_internal_function(n,k,d,beta,integrate,x):
    return P(n, k, x) * Wmatched(d, beta, x, integrate) * A(d, x)

def pf(d, n, k, beta=None, lb=None, ub=None, beta_and=False, integrate = True, prec=None):
    """
    Let `Pr[P_{k,n}]` be the probability that a popcount filter passes.  We assume the probability
    is over the vectors `u,v`. Let `¬G` be the event that two random vectors are not Gauss reduced.

    We start with Pr[P_{k,n}]::

        sage: pf(80, 128, 40)
        mpf('0.00031063713572376122')

        sage: pf(80, 128, 128)
        mpf('1.0000000000000002')

    Pr[P_{k,n} ∧ ¬G]::

        sage: pf(80, 128, 40, ub=mp.pi/3)
        mpf('3.3598092589552732e-7')

    Pr[¬G]::

        sage: pf(80, 128, 128, ub=mp.pi/3)
        mpf('1.0042233739846644e-6')

        sage: ngr_pf(80, 128, 128)
        mpf('1.0042233739846644e-6')

        sage: ngr(80)
        mpf('1.0042233739846629e-6')

    Pr[Pr_{k,n} ∧ G]::

        sage: pf(80, 128, 40, lb=mp.pi/3)
        mpf('0.00031030115479786595')

    Pr[G]::

        sage: pf(80, 128, 128, lb=mp.pi/3)
        mpf('0.99999899577662632')

        sage: gr_pf(80, 128, 128)
        mpf('0.99999899577662632')

        sage: gr(80)
        mpf('0.99999899577662599')

    Pr[P_{k,n} | C(w,β)]::

        sage: pf(80, 128, 40, beta=mp.pi/3)
        mpf('0.019786655048072234')

    Pr[P_{k,n}  ∧ ¬G | C(w,β)]::

        sage: pf(80, 128, 40, beta=mp.pi/3, ub=mp.pi/3)
        mpf('0.00077177364924089652')

    Pr[¬G | C(w,β)]::

        sage: pf(80, 128, 128, beta=mp.pi/3, ub=mp.pi/3)
        mpf('0.0021964683579090904')
        sage: ngr_pf(80, 128, 128, beta=mp.pi/3)
        mpf('0.0021964683579090904')
        sage: ngr(80, beta=mp.pi/3)
        mpf('0.0021964683579090904')

    Pr[Pr_{k,n} ∧ G | C(w,β)]::

        sage: pf(80, 128, 40, beta=mp.pi/3, lb=mp.pi/3)
        mpf('0.019014953591444488')

        sage: gr_pf(80, 128, 40, beta=mp.pi/3)
        mpf('0.019014953591444488')

    Pr[G | C(w,β)]::

        sage: pf(80, 128, 128, beta=mp.pi/3, lb=mp.pi/3)
        mpf('0.99780353164285229')
        sage: gr_pf(80, 128, 128, beta=mp.pi/3)
        mpf('0.99780353164285229')
        sage: gr(80, beta=mp.pi/3)
        mpf('0.9978035316420909')

    :param d: We consider the sphere `S^{d-1}`
    :param n: Number of popcount vectors
    :param k: popcount threshold
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param lb: lower bound of integration (see above)
    :param ub: upper bound of integration (see above)
    :param beta_and: return Pr[P_{k,n} ∧ C(w,β)] instead of Pr[P_{k,n} | C(w,β)]
    :param prec: compute with this precision

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        if lb is None:
            lb = 0
        if ub is None:
            ub = mp.pi
        if beta is None:
            return mp.quad(lambda x: P(n, k, x) * A(d, x), (lb, ub), error=True)[0]
        else:
            num = mp.quad(lambda x: P(n, k, x) * W(d, beta, beta, x, integrate) * A(d, x), (lb, min(ub, 2 * beta)), error=True)
            if not beta_and:
                den = mp.quad(lambda x: W(d, beta, beta, x, integrate) * A(d, x), (0, 2 * beta), error=True)
            else:
                den = [1,0]
            num_slices = 2
            high_guess = abs((num[0] + num[1])/(den[0] - den[1]))
            low_guess = (num[0] - num[1])/(den[0]+den[1])
            real_ub = min(ub,2*beta)
            while abs(high_guess - low_guess) > low_guess*1e-3:
                mp.prec += 2
                if DisplayConfig.display:
                    print("Computing false probability integral with",num_slices,"slices",end='\r')
                ncores = min(MultiProcessingConfig.num_cores,num_slices)
                jobs = []
                fs = [partial(pf_internal_function,n,k,d,beta,integrate),partial(ngr_internal_function,d,beta,integrate)]
                for core in range(ncores):
                    slices = list(range(num_slices))[core::ncores]
                    num_bounds = [(lb + i*(real_ub - lb)/num_slices, lb + (i+1)*(real_ub - lb)/num_slices) for i in slices]
                    den_bounds = [(i*2*beta/num_slices, (i+1)*2*beta/num_slices) for i in slices]
                    jobs.append((fs,[num_bounds,den_bounds]))
               

                # Create a Pool with the specified number of cores
                if ncores > 1:
                    with Pool(processes=ncores) as pool:
                        # Apply the parallel function to each chunk using the Pool
                        results = pool.map(multi_integrals,jobs)
                else:
                    results = [multi_integrals(job) for job in jobs]
                summed_results = [[0,0],[0,0]]
                for result in results: # iterate over processors
                    for i in range(2): #iterate over functions
                        for integral in result[i]: # iterate over sub-intervals
                            summed_results[i][0] += integral[0]
                            summed_results[i][1] += integral[1]
                num = summed_results[0]
                den = summed_results[1]                
                num_slices *= 2
            
                high_guess = (num[0] + num[1])/(den[0] - den[1] + num[0] - num[1])
                low_guess = (num[0] - num[1])/(den[0]+den[1]+num[0]+num[1])
            if DisplayConfig.display:
                print(' '*100,end='\r')
            return num[0] / den[0]


ngr_pf = partial(pf, lb=0, ub=mp.pi / 3)
gr_pf = partial(pf, lb=mp.pi / 3)



def ngr(d, beta=None, prec=None, kappangle = mp.pi / 3, integrate = True):
    """
    Probability that two random vectors (in a cap parameterised by β) are not Gauss reduced.
    SJ note: "Not Gauss reduced" seems to be the same as "at angle at most pi/3"; here it is 
    changed to "at angle at most arccos(kappa)"

    :param d: We consider the sphere `S^{d-1}`
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param prec: compute with this precision

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        if beta is None:
            return C(d, kappangle)
        elif beta < mp.acos(1 - mp.cos(kappangle)) / 2:
            return mp.mpf(1.0)
        else:
            # Pr[¬G ∧ E]
            # where E = "both vectors are in a bucket around some `w` of angle beta"
            # and G = "Gauss reduced" i.e. "at angle less than arccos(kappa)"
            num = mp.quad(lambda x: W(d, beta, beta, x, integrate) * A(d, x), (0, kappangle), error=True)
            # Pr[E]
            den = mp.quad(lambda x: W(d, beta, beta, x, integrate) * A(d, x), (kappangle, 2 * beta), error=True)
            num_slices = 2
            high_guess = (num[0] + num[1])/(den[0] - den[1] + num[0] - num[1])
            low_guess = (num[0] - num[1])/(den[0]+den[1]+num[0]+num[1])
            while abs(high_guess - low_guess) > low_guess*1e-3:
                mp.prec += 2
                if DisplayConfig.display:
                    print("Computing NGR integral with",num_slices,"slices",end='\r')
                ncores = min(MultiProcessingConfig.num_cores,num_slices)
                jobs = []
                fs = [partial(ngr_internal_function,d,beta,integrate)]*2
                for core in range(ncores):
                    slices = list(range(num_slices))[core::ncores]
                    num_bounds = [(i*kappangle/num_slices, (i+1)*kappangle/num_slices) for i in slices]
                    den_bounds = [(kappangle + i*(2*beta-kappangle)/num_slices, kappangle+(i+1)*(2*beta - kappangle)/num_slices) for i in slices]
                    jobs.append((fs,[num_bounds,den_bounds]))
               

                # Create a Pool with the specified number of cores
                if ncores > 1:
                    with Pool(processes=ncores) as pool:
                        # Apply the parallel function to each chunk using the Pool
                        results = pool.map(multi_integrals,jobs)
                else:
                    results = [multi_integrals(job) for job in jobs]
                summed_results = [[0,0],[0,0]]
                for result in results: # iterate over processors
                    for i in range(2): #iterate over functions
                        for integral in result[i]: # iterate over sub-intervals
                            summed_results[i][0] += integral[0]
                            summed_results[i][1] += integral[1]
                num = summed_results[0]
                den = summed_results[1]                
                num_slices *= 2
                high_guess = (num[0] + num[1])/(den[0] - den[1] + num[0] - num[1])
                low_guess = (num[0] - num[1])/(den[0]+den[1]+num[0]+num[1])
            den = den[0] + num[0]
            if DisplayConfig.display:
                print(' '*100,end='\r')
            # Pr[¬G | E] = Pr[¬G ∧ E]/Pr[E]
            return num[0] / den


def gr(d, beta=None, prec=None):
    """
    Probability that two random vectors (in a cap parameterised by β) are Gauss reduced.

    :param d: We consider the sphere `S^{d-1}`
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param prec: compute with this precision

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        return 1 - ngr(d, beta, prec)


def probabilities(d, n, k, beta=None, prec=None, kappangle=1/2):
    """
    Useful probabilities.

    :param d: We consider the sphere `S^{d-1}`
    :param n: Number of popcount vectors
    :param k: popcount threshold
    :param beta: If not ``None`` vectors are considered in a bucket around some `w` with angle β.
    :param prec: compute with this precision
    :param kappangle: threshold angle below which we consider two vectors reducing (not Gauss reduced?)

    """
    prec = prec if prec else mp.prec

    with mp.workprec(prec):
        pf_ = pf(d, n, k, beta=beta, prec=prec)
        ngr_ = ngr(d, beta=beta, prec=prec,kappangle=kappangle)
        ngr_pf_ = ngr_pf(d, n, k, beta=beta, prec=prec,ub=kappangle)
        gr_pf_ = gr_pf(d, n, k, beta=beta, prec=prec,lb=kappangle)
        rho = 1 - ngr_pf_ / pf_
        eta = 1 - ngr_pf_ / ngr_

        probs = Probabilities(
            d=d,
            n=n,
            k=k,
            ngr=ngr_,
            gr=1 - ngr_,
            pf=pf_,
            gr_pf=gr_pf_,
            ngr_pf=ngr_pf_,
            rho=rho,
            eta=eta,
            beta=beta,
            prec=prec,
        )
        return probs
