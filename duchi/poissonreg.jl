# poissonreg.jl
#
# Code for poisson regression.

using Distributions;

# (A, b, x) = GeneratePoissonData(m::Int64 = 200, n::Int64 = 20)
#
# Generates an m-by-n matrix A of measurement vectors a_i = A[i, :], where each
# a_i is uniform on the unit sphere. Chooses x uniformly from the sphere of
# radius sqrt(n), and draws b_i as
#
#   b_i ~ Poisson(exp(a_i' * x))
function GeneratePoissonData(m::Int64 = 200, n::Int64 = 20;
                             generate_gaussian::Bool = false,
                             condition_number::Float64 = 1.0)
  A = randn(m, n);
  if (!generate_gaussian)
    A = A ./ sqrt.(repeat(sum(A.^2, dims=2), outer = (1, n)));
  end
  if (condition_number > 1.0)
    # Make A have condition number given
    condition_vec = Vector(LinRange(1.0, condition_number, n));
    condition_vec = condition_vec / norm(condition_vec);
    A = A .* repeat(condition_vec', outer = (m, 1));
  end
  x = randn(n);
  x = sqrt(n) * x / norm(x);
  b = zeros(Int64, m);
  for ii = 1:m
    R = Poisson(exp(dot(A[ii, :], x)));
    b[ii] = rand(R);
  end
  return (A, b, x);
end

# PoissonObj(A::Matrix{Float64}, b::Vector{Int64}, x::Vector{Float64})
#
# Computes the (average) log loss for the poisson regression problem,
# where we recall that the log-likelihood for b ~ Poisson(lambda) is
#
#  p(b | lambda) = exp(-lambda) * lambda^b / b!
#
# For a data vector a, the parameter lambda = exp(a' * x), so we have log loss
# (ignoring the factorial term)
#
# l(x; (a, b)) = -b * dot(a, x) + exp(dot(a, x)).
function PoissonObj(A::Matrix{Float64}, b::Vector{Int64}, x::Vector{Float64})
  mm = size(A, 1);
  objective = -sum(b .* (A * x)) + sum(exp.(A*x)); # no need to add b!
  return objective / mm;
end

# g = FullPoissonGradient(A, b, x)
#
# Sets g to be the gradient of the poisson objective and returns it.
function FullPoissonGradient(A::Matrix{Float64}, b::Vector{Int64},
                             x::Vector{Float64})
  mm = size(A, 1);
  grad = (-A' * b + A' * exp.(A * x)) / mm;
  return grad;
end

# x = ProximalUpdatePoisson(a::Vector{Float64}, b::Float64,
#                              x_init::Vector, stepsize::Float64; accuracy)
#
# Sets x to minimize
#
#  -b * a' * x + exp(x' * a) + norm(x - x_init)^2 / (2 * stepsize).
#
# Returns the minimizing x. The accuracy parameter governs the
# accuracy of the solution: we recommend a line search.
function ProximalUpdatePoisson(a::Vector{Float64}, b::Int64,
                               x_init::Vector{Float64}, stepsize::Float64;
                               eps_accuracy::Float64 = 1e-10)
  # Replace this code to return the correct update
  return x_init;
end

# x = TruncatedUpdatePoisson(a::Vector{Float64}, b::Int64,
#                            x_init::Vector{Float64}, stepsize::Float64)
#
# Let l(x; (a, b)) = -b * a' * x + exp(a' * x). Sets x to minimize the
# lower-bounded approximation to F at x_init, that is, for
#
#  l_approx(x) = max{ l(x_init) + l'(x_init) * (x - x_init)),
#                     inf_x l(x; (a, b)) }
#
# sets x to minimize
#
#  l_approx(x) + norm(x - x_init)^2 / (2 * stepsize).
#
# Returns the minimizing x.
function TruncatedUpdatePoisson(a::Vector{Float64}, b::Int64,
                                x_init::Vector{Float64}, stepsize::Float64)
  # Replace this code to return the correct update
  return x_init;
end

# x = SGUpdatePoisson(a::Vector{Float64}, b::Int64,
#                     x_init::Vector{Float64}, stepsize::Float64)
#
# Computes the stochastic gradient update for Poisson regression.
function SGUpdatePoisson(a::Vector{Float64}, b::Int64,
                         x_init::Vector{Float64}, stepsize::Float64)
  # Replace this code to return the correct update
  return x_init;
end
