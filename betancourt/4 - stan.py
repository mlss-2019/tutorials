############################################################
# Initial setup
############################################################

import math

import matplotlib
import matplotlib.pyplot as plot

import pystan
import stan_utility

help(stan_utility)

light="#DCBCBC"
light_highlight="#C79999"
mid="#B97C7C"
mid_highlight="#A25050"
dark="#8F2727"
dark_highlight="#7C0000"
green="#00FF00"

# To facilitate the computation of Markov chain Monte Carlo estimators 
# let's define a _Welford accumulator_ that computes empirical summaries
# of a sample in a single pass
def welford_summary(x, L = 100):
  summary = [0] * (L + 1)
  for n in range(len(x)):
    delta = x[n] - summary[0]
    summary[0] += delta / (n + 1)
    for l in range(L):
      if n > l:
        summary[l + 1] += delta * (x[n - l] - summary[0])

  norm = 1.0 / (len(x) - 1)
  for l in range(L): summary[l + 1] *= norm
  return summary

# We can then use the Welford accumulator output to compute the
# Markov chain Monte Carlo estimators and their properties
def compute_mcmc_stats(x, L = 20):
  summary = welford_summary(x, L)
  
  mean = summary[0]
  var = summary[1]
  acov = summary[1:(L + 1)]
  
  # Compute the effective sample size
  rho_hat_s = [0] * L
  rho_hat_s[1] = acov[1] / var
  
  # First we transform our autocovariances into Geyer's initial positive sequence
  max_s = 1
  for s in [ 2 * i + 1 for i in range((L - 1) / 2) ]:
    rho_hat_even = acov[s + 1] / var
    rho_hat_odd = acov[s + 2] / var;
    
    max_s = s + 2  
    
    if rho_hat_even + rho_hat_odd > 0:
      rho_hat_s[s + 1] = rho_hat_even
      rho_hat_s[s + 2] = rho_hat_odd
    else:   
      break
  
  # Then we transform this output into Geyer's initial monotone sequence
  for s in [ 2 * i + 3 for i in range((max_s - 2)/ 2) ]:
    if rho_hat_s[s + 1] + rho_hat_s[s + 2] > rho_hat_s[s - 1] + rho_hat_s[s]:
      rho_hat_s[s + 1] = 0.5 * (rho_hat_s[s - 1] + rho_hat_s[s])
      rho_hat_s[s + 2] = rho_hat_s[s + 1]
  
  ess = len(x) / (1.0 + 2 * sum(rho_hat_s))
  
  return [mean, math.sqrt(var / ess), math.sqrt(var), ess]

def compute_running_estimator(x):
  N = len(x)
  stride = 50
  M = N / stride

  iters = [ stride * (i + 1) for i in range(M) ]
  
  x1_mean = [0] * M 
  x1_se = [0] * M

  for m in range(M):
    running_samples = x0[0:iters[m]]
    mcmc_stats = compute_mcmc_stats(running_samples)
    x1_mean[m] = mcmc_stats[0]
    x1_se[m] = mcmc_stats[1]
    
  return iters, x1_mean, x1_se

############################################################
# Normal Model
############################################################

# Compile Stan program and fit with dynamic Hamiltonian Monte Carlo
model = stan_utility.compile_model('normal.stan')
fit = model.sampling(seed=4838282)

# Check diagnostics
stan_utility.check_all_diagnostics(fit)

# Check MCMC estimators
print(fit)

############################################################
# Student-t Model
############################################################

model = stan_utility.compile_model('student_t.stan')

# 100 degrees of freedom
data = dict(nu = 100)
fit100 = model.sampling(data=data, seed=4838282,
                        control=dict(metric="unit_e", stepsize=0.7, adapt_engaged=False))
stan_utility.check_all_diagnostics(fit100)

sampler_params = fit100.get_sampler_params(inc_warmup=False)
stepsizes100 = [sampler_params[n]['stepsize__'][0] 
                for n in range(4) ]
times100 = [stepsizes100[n] * x for n in range(4)
          for x in sampler_params[n]['n_leapfrog__'] ]
energies100 = [x for y in sampler_params for x in y['energy__']]

x0 = fit100.extract(permuted=False)[:,:,0].flatten()
iters, x1_mean, x1_se = compute_running_estimator(x0)

plot.fill_between(iters, 
                  [ x1_mean[m] - 2 * x1_se[m] for m in range(len(iters)) ],
                  [ x1_mean[m] + 2 * x1_se[m] for m in range(len(iters)) ],
                  facecolor=light, color=light)
plot.plot(iters, x1_mean, color=dark)
plot.plot([iters[0], iters[-1]], [0, 0], color='grey', linestyle='--')

plot.gca().set_xlim([0, 4000])
plot.gca().set_xlabel("Iteration")
plot.gca().set_ylim([-0.5, 0.5])
plot.gca().set_ylabel("Monte Carlo Estimator")

plot.show()

# 5 degrees of freedom
data = dict(nu = 5)
fit5 = model.sampling(data=data, seed=4838282, 
                      control=dict(metric="unit_e", stepsize=0.7, adapt_engaged=False))
stan_utility.check_all_diagnostics(fit5)

sampler_params = fit5.get_sampler_params(inc_warmup=False)
stepsizes5 = [sampler_params[n]['stepsize__'][0] 
              for n in range(4) ]
times5 = [stepsizes5[n] * x for n in range(4)
          for x in sampler_params[n]['n_leapfrog__'] ]
energies5 = [x for y in sampler_params for x in y['energy__']]

# 2 degrees of freedom -- no more component variances!
data = dict(nu = 2)
fit2 = model.sampling(data=data, seed=4838282,
                      control=dict(metric="unit_e", stepsize=0.7, adapt_engaged=False))
stan_utility.check_all_diagnostics(fit2)

sampler_params = fit2.get_sampler_params(inc_warmup=False)
stepsizes2 = [sampler_params[n]['stepsize__'][0] 
              for n in range(4) ]
times2 = [stepsizes2[n] * x for n in range(4)
          for x in sampler_params[n]['n_leapfrog__'] ]
energies2 = [x for y in sampler_params for x in y['energy__']]

# 1 degree of freedom -- no more component means or variances!
data = dict(nu = 1)
fit1 = model.sampling(data=data, seed=4838282,
                      control=dict(metric="unit_e", stepsize=0.7, 
                                   max_treedepth=12, adapt_engaged=False))
stan_utility.check_all_diagnostics(fit1, 12)

sampler_params = fit1.get_sampler_params(inc_warmup=False)
stepsizes1 = [sampler_params[n]['stepsize__'][0] 
              for n in range(4) ]
times1 = [stepsizes1[n] * x for n in range(4)
          for x in sampler_params[n]['n_leapfrog__'] ]
energies1 = [x for y in sampler_params for x in y['energy__']]

x0 = fit1.extract(permuted=False)[:,:,0].flatten()
iters, x1_mean, x1_se = compute_running_estimator(x0)

plot.fill_between(iters, 
                  [ x1_mean[m] - 2 * x1_se[m] for m in range(len(iters)) ],
                  [ x1_mean[m] + 2 * x1_se[m] for m in range(len(iters)) ],
                  facecolor=light, color=light)
plot.plot(iters, x1_mean, color=dark)
plot.plot([iters[0], iters[-1]], [0, 0], color='grey', linestyle='--')

plot.gca().set_xlim([0, 4000])
plot.gca().set_xlabel("Iteration")
plot.gca().set_ylim([-50, 50])
plot.gca().set_ylabel("Monte Carlo Estimator")

plot.show()

# Plot comparison of integration time scalings
f, axarr = plot.subplots(1, 4)
    
axarr[0].set_title("nu = 100")
axarr[0].scatter(energies100, [math.log(x) for x in times100], color=dark, alpha=0.1)
axarr[0].set_xlim(0, 50)
axarr[0].set_xlabel("Energy")
axarr[0].set_ylim([0, 7])
axarr[0].set_ylabel("Log Integration Time")

axarr[1].set_title("nu = 5")
axarr[1].scatter(energies5, [math.log(x) for x in times5], color=dark, alpha=0.1)
axarr[1].set_xlim(0, 50)
axarr[1].set_xlabel("Energy")
axarr[1].set_ylim([0, 7])

axarr[2].set_title("nu = 2")
axarr[2].scatter(energies2, [math.log(x) for x in times2], color=dark, alpha=0.1)
axarr[2].set_xlim(0, 50)
axarr[2].set_xlabel("Energy")
axarr[2].set_ylim([0, 7])

axarr[3].set_title("nu = 1")
axarr[3].scatter(energies1, [math.log(x) for x in times1], color=dark, alpha=0.1)
axarr[3].set_xlim(0, 50)
axarr[3].set_xlabel("Energy")
axarr[3].set_ylim([0, 7])

plot.subplots_adjust(wspace=0.5)
plot.show()

############################################################
# Bayesian Inference
############################################################

# Let's consider an example in the context of Bayesian inference!

# We first simulate an observation and save it to a file
N = 1
simu_data = dict(N = N)

simu_model = stan_utility.compile_model('simulate_data.stan')
simu = simu_model.sampling(data=simu_data, iter=1, chains=1, seed=4838282,
                           algorithm="Fixed_param")

data = dict(N = N, y = simu.extract()['y'].flatten())
pystan.stan_rdump(data, 'simulation.data.R')

# Now we can read that data back in and use Hamiltonian
# Monte Carlo to estimate posterior expectation values
input_data = pystan.read_rdump('simulation.data.R')

model = stan_utility.compile_model('fit_data.stan')
fit = model.sampling(data=input_data, seed=4938483)

# Check diagnostics
stan_utility.check_all_diagnostics(fit)

# That doesn't look good.  Let's investigate the divergent
# samples in the context of the non-divergent samples to
# see what's going on.
nondiv_params, div_params = stan_utility.partition_div(fit)

plot.scatter(nondiv_params['mu'],
             [math.log(x) for x in nondiv_params['sigma']],
              color = mid_highlight, alpha=0.05)
plot.scatter(div_params['mu'],
             [math.log(x) for x in div_params['sigma']],
              color = green, alpha=0.5)

plot.gca().set_xlabel("mu")
plot.gca().set_ylabel("sigma")
plot.show()
