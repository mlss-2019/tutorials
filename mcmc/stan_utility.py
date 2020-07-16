############################################################
# Copyright 2019 Michael Betancourt
# Licensed under the new BSD (3-clause) license:
#
# https://opensource.org/licenses/BSD-3-Clause
############################################################

import pystan
import pickle
import numpy


def check_div(fit, quiet=False):
    """Check transitions that ended with a divergence"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    divergent = [x for y in sampler_params for x in y['divergent__']]
    n = sum(divergent)
    N = len(divergent)

    if not quiet:
        print('{} of {} iterations ended with a divergence ({}%)'.format(n, N, 100 * n / N))

    if n > 0:
        if not quiet:
            print('  Try running with larger adapt_delta to remove the divergences')
        else:
            return False
    else:
        if quiet:
            return True


def check_treedepth(fit, max_treedepth=10, quiet=False):
    """Check transitions that ended prematurely due to maximum tree depth limit"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    depths = [x for y in sampler_params for x in y['treedepth__']]
    n = sum(1 for x in depths if x == max_treedepth)
    N = len(depths)

    if not quiet:
        print(('{} of {} iterations saturated the maximum tree depth of {}'
              + ' ({}%)').format(n, N, max_treedepth, 100 * n / N))
    if n > 0:
        if not quiet:
            print('  Run again with max_treedepth set to a larger value to avoid saturation')
        else:
            return False
    else:
        if quiet:
            return True


def check_energy(fit, quiet=False):
    """Checks the energy fraction of missing information (E-FMI)"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    no_warning = True
    for chain_num, s in enumerate(sampler_params):
        energies = s['energy__']
        numer = sum((energies[i] - energies[i - 1])**2 for i in range(1, len(energies))) / len(energies)
        denom = numpy.var(energies)
        if numer / denom < 0.2:
            if not quiet:
                print('Chain {}: E-BFMI = {}'.format(chain_num, numer / denom))
            no_warning = False

    if no_warning:
        if not quiet:
            print('E-BFMI indicated no pathological behavior')
        else:
            return True
    else:
        if not quiet:
            print('  E-BFMI below 0.2 indicates you may need to reparameterize your model')
        else:
            return False


def check_n_eff(fit, quiet=False):
    """Checks the effective sample size per iteration"""
    fit_summary = fit.summary(probs=[0.5])
    n_effs = [x[4] for x in fit_summary['summary']]
    names = fit_summary['summary_rownames']
    n_iter = len(fit.extract()['lp__'])

    no_warning = True
    for n_eff, name in zip(n_effs, names):
        ratio = n_eff / n_iter
        if ratio < 0.001:
            if not quiet:
                print('n_eff / iter for parameter {} is {}!'.format(name, ratio))
            no_warning = False

    if no_warning:
        if not quiet:
            print('n_eff / iter looks reasonable for all parameters')
        else:
            return True
    else:
        if not quiet:
            print('  n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated')
        else:
            return False


def check_rhat(fit, quiet=False):
    """Checks the potential scale reduction factors"""
    from math import isnan
    from math import isinf

    fit_summary = fit.summary(probs=[0.5])
    rhats = [x[5] for x in fit_summary['summary']]
    names = fit_summary['summary_rownames']

    no_warning = True
    for rhat, name in zip(rhats, names):
        if rhat > 1.1 or isnan(rhat) or isinf(rhat):
            if not quiet:
                print('Rhat for parameter {} is {}!'.format(name, rhat))
            no_warning = False
    if no_warning:
        if not quiet:
            print('Rhat looks reasonable for all parameters')
        else:
            return True
    else:
        if not quiet:
            print('  Rhat above 1.1 indicates that the chains very likely have not mixed')
        else:
            return False


def check_all_diagnostics(fit, max_treedepth=10, quiet=False):
    """Checks all MCMC diagnostics"""

    if not quiet:
        check_n_eff(fit)
        check_rhat(fit)
        check_div(fit)
        check_treedepth(fit, max_treedepth=max_treedepth)
        check_energy(fit)
    else:
        warning_code = 0
        if not check_n_eff(fit, quiet):
            warning_code = warning_code | (1 << 0)
            print(warning_code)
        if not check_rhat(fit, quiet):
            warning_code = warning_code | (1 << 1)
        if not check_div(fit, quiet):
            warning_code = warning_code | (1 << 2)
        if not check_treedepth(fit, max_treedepth, quiet):
            warning_code = warning_code | (1 << 3)
        if not check_energy(fit, quiet):
            warning_code = warning_code | (1 << 4)

        return warning_code


def parse_warning_code(warning_code):
    """Parses warning code into individual failures"""
    if warning_code & (1 << 0):
        print("n_eff / iteration warning")
    if warning_code & (1 << 1):
        print("rhat warning")
    if warning_code & (1 << 2):
        print("divergence warning")
    if warning_code & (1 << 3):
        print("treedepth warning")
    if warning_code & (1 << 4):
        print("energy warning")


def _by_chain(unpermuted_extraction):
    num_chains = len(unpermuted_extraction[0])
    result = [[] for _ in range(num_chains)]
    for c in range(num_chains):
        for i in range(len(unpermuted_extraction)):
            result[c].append(unpermuted_extraction[i][c])
    return numpy.array(result)


def _shaped_ordered_params(fit):
    ef = fit.extract(permuted=False, inc_warmup=False)  # flattened, unpermuted, by (iteration, chain)
    ef = _by_chain(ef)
    ef = ef.reshape(-1, len(ef[0][0]))
    ef = ef[:, 0:len(fit.flatnames)]  # drop lp__
    shaped = {}
    idx = 0
    for dim, param_name in zip(fit.par_dims, fit.extract().keys()):
        length = int(numpy.prod(dim))
        shaped[param_name] = ef[:, idx:idx + length]
        shaped[param_name].reshape(*([-1] + dim))
        idx += length
    return shaped


def partition_div(fit):
    """ Returns parameter arrays separated into divergent and non-divergent transitions"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    div = numpy.concatenate([x['divergent__'] for x in
                            sampler_params]).astype('int')
    params = _shaped_ordered_params(fit)
    nondiv_params = dict((key, params[key][div == 0]) for key in params)
    div_params = dict((key, params[key][div == 1]) for key in params)
    return nondiv_params, div_params


def compile_model(filename, model_name=None):
    """This will automatically cache models - great if you're just running a
    script on the command line.

    See http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html"""
    from hashlib import md5

    with open(filename) as f:
        model_code = f.read()
        code_hash = md5(model_code.encode('ascii')).hexdigest()
        if model_name is None:
            cache_fn = 'cached-model-{}.pkl'.format(code_hash)
        else:
            cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
        try:
            sm = pickle.load(open(cache_fn, 'rb'))
        except:
            sm = pystan.StanModel(model_code=model_code)
            with open(cache_fn, 'wb') as f1:
                pickle.dump(sm, f1)
        else:
            print("Using cached StanModel")
        return sm
