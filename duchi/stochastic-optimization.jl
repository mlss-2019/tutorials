# stochastic-optimization.jl
#
# Wrapper methods for stochastic optimization problems.

module StochOpt

using LinearAlgebra;
using Distributions;

include("linreg.jl");
include("poissonreg.jl");
include("phase-retrieval.jl");
include("utilities.jl");
include("logreg.jl");

# Update types are
#
#  :proximal  - Full proximal update
#  :truncated  - Truncated update
#  :sgm  -  Stochastic gradient method (linear approximation) update
#
# The problem types are
#
#  :linreg - Linear regression
#  :logreg - Logistic regression
#  :poisson - Poisson regression
#  :phase - Phase retrieval

# obj_gaps =
#   MultiStepsizeExperiment(problem_type, update_type; n_tests, n_sample,
#                           dim, epsilons, stepsizes, maxiter, condition,
#                           noise)
#
# Performs a number of experiments to determine how sensitive problems
# of type problem_type are to initial stepsizes for the given
# update_type. Returns a tensor of size
#
#  num_computed_objectives -by- length(stepsizes) -by- n_tests
#
# where the entry [i,j,k] in the tensor corresponds to the obective
# value gap at "iteration" i of the given method, during experiment k,
# using initial stepsize stepsizes[j].
#
# The input data is as follows:
#
#  problem_type - one of the problem types above
#  update_type - type of optimizer: :sgm, :proximal, :truncated.
#  n_tests - number of experiments to run
#  n_sample - sample size for individual experiments
#  dim - dimension of problem to solve
#        Note that for matrix completion, the dimension is taken to be
#        the *rank* of the desired solution.
#  stepsizes - Different stepsizes to use
#  maxiter - maximum number of iterations
#  noise - magnitude of noise to add
#  condition - condition number of data matrix
function MultiStepsizeExperiment(
  problem_type::Symbol = :linreg,
  update_type::Symbol = :proximal;
  n_tests::Int64 = 20,
  n_sample::Int64 = 400,
  dim::Int64 = 40,
  stepsizes = logspace(-1.5, log10(100), 13),
  step_pow::Float64 = 0.6,
  maxiter::Int64 = 20 * n_sample,
  compute_every::Int64 = floor(Int64, maxiter / 100),
  condition::Float64 = 1.0,
  noise::Float64 = 0.0)
  
  # First, get the matrices of different times to epsilon.
  num_stepsizes = length(stepsizes);
  objectives = zeros(floor(Int64, maxiter / compute_every) + 1,
                     num_stepsizes, n_tests);
  # Now iterate over number of tests.
  for test_ind = 1:n_tests
    println("*** Experiment ", test_ind, " of ", n_tests, " ***");

    opt_objval = 0.0;
    (A, b, x_opt) = GenerateData(problem_type, n_sample, dim, noise, condition);
    minimal_value = FindMinimalValue(problem_type, A, b, quiet=true);
    for step_ind = 1:length(stepsizes)
      init_step = stepsizes[step_ind];
      println("\tInit step = ", init_step);
      x_init = zeros(size(A, 2));
      if (problem_type == :phase)
        x_init = randn(size(A, 2)) / 2;
      end
      obj_vals = StochasticProxPoint(A, b;
                                     problem_type = problem_type,
                                     update_type = update_type,
                                     init_stepsize = init_step,
                                     step_pow = step_pow,
                                     maxiter = maxiter,
                                     compute_every = compute_every,
                                     x_init = x_init);
      # remove NaNs
      obj_vals[isnan.(obj_vals)] .= Inf;
      objectives[:, step_ind, test_ind] = obj_vals .- minimal_value;
    end
  end
  return objectives;
end

# obj_gaps =
#   MultiMinibatchExperiment(problem_type, update_type; n_tests, n_sample,
#                           dim, epsilons, stepsize, maxiter, condition,
#                           noise, minibatch_sizes)
#
# Performs a number of experiments to determine how sensitive problems
# of type problem_type are to initial minibatch size for the given
# update_type. Returns a tensor of size
#
#  num_computed_objectives -by- length(minibatch_sizes) -by- n_tests
#
# where the entry [i,j,k] in the tensor corresponds to the obective
# value gap at "iteration" i of the given method, during experiment k,
# using minibatch size minibatch_sizes[j].
#
# The input data is as follows:
#
#  problem_type - one of the problem types above
#  update_type - type of optimizer: :sgm, :proximal, :truncated.
#  n_tests - number of experiments to run
#  n_sample - sample size for individual experiments
#  dim - dimension of problem to solve
#        Note that for matrix completion, the dimension is taken to be
#        the *rank* of the desired solution.
#  stepsize - Initial stepsize to use
#  maxiter - maximum number of iterations
#  noise - magnitude of noise to add
#  condition - condition number of data matrix
function MultiMinibatchExperiment(
  problem_type::Symbol = :linreg,
  update_type::Symbol = :proximal;
  n_tests::Int64 = 20,
  n_sample::Int64 = 400,
  dim::Int64 = 40,
  stepsize::Float64 = 1.0,
  minibatch_sizes::Array{Int64} = [1, 2, 4, 8, 16, 32, 64],
  step_pow::Float64 = 0.6,
  maxiter::Int64 = 20 * n_sample,
  compute_every::Int64 = floor(Int64, maxiter / 100),
  condition::Float64 = 1.0,
  noise::Float64 = 0.0)
  
  # First, get the matrices of different times to epsilon.
  num_minibatches = length(minibatch_sizes);
  objectives = zeros(floor(Int64, maxiter / compute_every) + 1,
                     num_minibatches, n_tests);
  # Now iterate over number of tests.
  for test_ind = 1:n_tests
    println("*** Experiment ", test_ind, " of ", n_tests, " ***");

    opt_objval = 0.0;
    (A, b, x_opt) = GenerateData(problem_type, n_sample, dim, noise, condition);
    minimal_value = FindMinimalValue(problem_type, A, b, quiet=true);
    for minibatch_ind = 1:length(minibatch_sizes)
      x_init = zeros(size(A, 2));
      if (problem_type == :phase)
        x_init = randn(size(A, 2)) / 2;
      end
      minibatch_size = minibatch_sizes[minibatch_ind];
      println("Minibatch size = ", minibatch_size);
      init_stepsize = stepsize * sqrt(minibatch_size);
      obj_vals = StochasticProxPoint(A, b;
                                     problem_type = problem_type,
                                     update_type = update_type,
                                     init_stepsize = init_stepsize,
                                     minibatch_size = minibatch_size,
                                     step_pow = step_pow,
                                     maxiter = maxiter,
                                     compute_every = compute_every,
                                     x_init = x_init);
      # remove NaNs
      obj_vals[isnan.(obj_vals)] .= Inf;
      objectives[:, minibatch_ind, test_ind] = obj_vals .- minimal_value;
    end
  end
  return objectives;
end

# objectives = StochasticProxPoint(A, b;
#                                  problem_type, update_type,
#                                  init_stepsize, step_pow,
#                                  maxiter,
#                                  compute_every, x_init, minibatch_size)
#
# Performs a stochastic model-based minimization on data (A, b),
# beginning from initial point x_init.
#
# Returns a vector of (estimated) *gaps* to optimality in the
# objective.
function StochasticProxPoint(A::Matrix{Float64}, b::Vector;
                             problem_type::Symbol = :linreg,
                             update_type::Symbol = :proximal,
                             init_stepsize::Float64 = 1.0,
                             step_pow::Float64 = .5,
                             maxiter::Int64 = 20 * size(A, 1),
                             compute_every::Int64 = floor(Int64,
                                                          maxiter / 100),
                             x_init::Vector{Float64} = zeros(size(A, 2)),
                             minibatch_size::Int64 = 1)
  # First, get the optimizers and updaters
  (objective, updater) = GetUpdateAndObjective(problem_type, update_type);
  (mm, nn) = size(A);
  if (length(b) != mm)
    error("Mismatched data sizes");
  end
  # Now, run the stochastic model-based minimization method.
  objectives = zeros(floor(Int64, maxiter / compute_every) + 1);
  x = x_init;
  objectives[1] = objective(A, b, x);
  num_objs_computed = 1;
  ind = 1;
  for iter = 1:maxiter
    # Choose a random index
    stepsize = init_stepsize / (iter^step_pow);
    if (minibatch_size > 1)
      inds = ceil.(Int64, mm * rand(minibatch_size));
      x = updater(@view(A[inds, :]), @view(b[inds]), x, stepsize);
    else
      ind = ceil(Int64, mm * rand())
      x = updater(A[ind, :], b[ind], x, stepsize);
    end
    if (mod(iter, compute_every) == 0)
      num_objs_computed += 1;
      objectives[num_objs_computed] = objective(A, b, x);
    end
  end
  return objectives;
end
  
# (obj, updater) = GetUpdateAndObjective(problem_type::Symbol,
#                                        update_type::Symbol)
#
# Given the symbols for problem type and update type, returns an
# objective function and an update function. The semantics of each of
# these is as follows. See above for lists of problem types and update types.
#
# The obj function takes as arguments (A, b, x), where A is an m-by-n
# data matrix, b is an m-dimensional label (or measurement or target)
# vector, and x is an n-dimensional parameter being optimized over.
# For example, if the problem_type is :linreg, then we would have
#
#  obj(A, b, x) = norm(A * x - b)^2 / (2 * m)
#
# where (m, n) = size(A).
#
# The updater function takes four arguments and computes a model-based
# minimization update. In particular, if the stochastic optimization
# problem being solved is to minimize
#
#  f(x) = (1/m) sum_{i=1}^m F(x; A[i, :], b[i])
#
# the updater takes as an argument (a, b, x_0, alpha), where a is a
# single data vector (i.e. one row of the matrix A), b a single label,
# x the current point, and alpha the stepsize, and returns
#
#  x = updater(a, b, x_0, alpha)
#    = argmin_x { F_{x_0}(x; a, b) + norm(x - x_0)^2 / (2 * alpha) }
#
# This is the model based update.
function GetUpdateAndObjective(problem_type::Symbol,
                               update_type::Symbol)
  objective = LinearRegressionObj;
  updater = ZeroUpdate;
  if (problem_type == :linreg)
    objective = LinearRegressionObj;
    if (update_type == :proximal)
      updater = ProximalUpdateLinreg;
    elseif (update_type == :truncated)
      updater = TruncatedUpdateLinreg;
    elseif (update_type == :sgm)
      updater = SGUpdateLinreg;
    else
      error("Unknown update type ", update_type);
    end
  elseif (problem_type == :poisson)
    objective = PoissonObj;
    if (update_type == :proximal)
      updater = ProximalUpdatePoisson;
    elseif (update_type == :truncated)
      updater = TruncatedUpdatePoisson;
    elseif (update_type == :sgm)
      updater = SGUpdatePoisson;
    else
      error("Unknown update type ", update_type);
    end
  elseif (problem_type == :phase)
    objective = PhaseRetrievalObj;
    if (update_type == :proximal)
      updater = ProximalUpdatePhaseRetrieval;
    elseif (update_type == :truncated)
      updater = TruncatedUpdatePhaseRetrieval;
    elseif (update_type == :sgm)
      updater = SGUpdatePhaseRetrieval;
    else
      error("Unknown update type ", update_type);
    end
  elseif (problem_type == :logreg)
    objective = LogisticRegressionObj;
    if (update_type == :proximal)
      error("No proximal update for logistic regression");
    elseif (update_type == :truncated)
      updater = TruncatedUpdateLogistic;
    elseif (update_type == :sgm)
      updater = SGUpdateLogistic;
    else
      error("Unknown update type ", update_type);
    end
  else
    error("Unknown problem type ", problem_type);
  end
  return (objective, updater);
end

# (A, b, x) = GenerateData(problem_type::Symbol,
#                          mm::Int64, nn::Int64,
#                          noise::Float64, condition::Float64)
#
# Generates data matrix A, target vector b, and optimal point x for
# the given problem type. The data matrix A is of size m-by-n, b of
# size m, and x of size n.
function GenerateData(problem_type::Symbol,
                      mm::Int64, nn::Int64,
                      noise::Float64, condition::Float64)
  if (problem_type == :linreg)
    return GenerateLinearRegressionData(mm, nn,
                                        condition_number = condition,
                                        noise = noise);
  elseif (problem_type == :poisson)
    return GeneratePoissonData(mm, nn, condition_number = condition);
  elseif (problem_type == :phase)
    return GeneratePhaseRetrievalData(mm, nn, condition_number = condition);
  elseif (problem_type == :logreg)
    return GenerateLogisticData(mm, nn, condition_number = condition,
                                snr = min(1 / noise, 100.0));
  end
  error("Data generation not implemented for problem type ", problem_type);
  return 0;
end

# Default zero objective
function ZeroObjective(A::Matrix{Float64}, b::Real,
                       x::Vector{Float64})
  return 0;
end

# Default zero update just returns input x.
function ZeroUpdate(a::Vector{Float64}, b::Real,
                    x::Vector{Float64}, stepsize::Float64)
  return x;
end

# FindMinimalValue(problem_type::Symbol,
#                  A::Matrix{Float64}, b::Vector)
#
# Performs a gradient descent method with backtracking line search
# (unless there is a known closed-form solution) to give the minimal value
# of the specified problem.
function FindMinimalValue(problem_type::Symbol,
                          A::Matrix{Float64}, b::Vector;
                          quiet::Bool = true)
  (mm, nn) = size(A);
  if (!quiet)
    println("Finding minimal value for problem of type ", problem_type);
  end
  if (problem_type == :linreg)
    if (mm <= nn)
      return 0;
    end
    x_opt = A\b;
    return norm(A * x_opt - b)^2 / (2 * mm);
  elseif (problem_type == :poisson)
    # Do a gradient method with backtracking line search
    return GradientDescentToValue(A, b,
                                  x -> PoissonObj(A, b, x),
                                  x -> FullPoissonGradient(A, b, x));
  elseif (problem_type == :logreg)
    # Do a gradient method with backtracking line search
    return GradientDescentToValue(A, b,
                                  x -> LogisticRegressionObj(A, b, x),
                                  x -> LogisticRegressionGrad(A, b, x));
  elseif (problem_type == :phase || problem_type == :matrix)
    # The minimal value for these problems--as there is a true,
    # consistent solution--is 0.
    return 0.0;
  else
    error("Unknown problem type ", problem_type);
  end
end

# obj = GradientDescentToValue(A::Matrix{Float64}, b::Vector,
#                              F::Function, Grad::Function;
#                              x_init = zeros(size(A, 2)),
#                              tol::Float64 = 1e-5)
#
# Performs gradient descent with a backtracking (Armijo) line search
# on the given objective F with gradient Grad until the gradient norm
# is less than tol.  Returns the value of the function F at termination.
function GradientDescentToValue(A::Matrix{Float64}, b::Vector,
                                F::Function, Grad::Function;
                                x_init = zeros(size(A, 2)),
                                tol::Float64 = 1e-5,
                                quiet::Bool = true)
  x = x_init;
  curr_obj = F(x);
  alpha = .25;  # Multiplier for sufficient decrease
  beta = .5;  # Multiplier for stepsizes
  stepsize = 1.0;
  grad_norm = Inf;
  curr_iter = 0;
  while (grad_norm > tol)
    g = Grad(x);
    grad_norm = norm(g);
    x_next = x - stepsize * g;
    guess_next_obj = curr_obj + alpha * dot(g, x_next - x);
    next_obj = F(x_next);
    while (guess_next_obj < next_obj)
      stepsize = stepsize * beta;
      x_next = x - stepsize * g;
      guess_next_obj = curr_obj + alpha * dot(g, x_next - x);
      next_obj = F(x_next);
    end
    curr_obj = next_obj;
    x = x_next;
    stepsize = min(stepsize / beta, 1.0);
    if (mod(curr_iter, 100) == 0 && !quiet)
      println("\tGradient descent: iteration ", curr_iter,
              ". Gradient norm: ", grad_norm);
    end
    curr_iter += 1;
  end
  return curr_obj;
end

end  # module StochOpt
