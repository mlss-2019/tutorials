# linreg.jl
#
# File with methods related to linear regression.
#
# Author: John Duchi (jduchi@stanford.edu)

# (A, b, x) = GenerateLinearRegressionData(m, n; condition_number, noise)
#
# Generates data for a linear regression problem
function GenerateLinearRegressionData(mm::Int64, nn::Int64;
                                      condition_number::Float64 = 1.0,
                                      noise::Float64 = 0.0)
  A = randn(mm, nn);
  x = randn(nn);
  if (condition_number > 1)
    D = LinRange(1, condition_number, nn);
    A = A * Diagonal(D);
  end
  b = A * x + noise * randn(mm);
  return (A, b, x);
end

# o = LinearRegressionObj(A::Matrix{Float64}, b::Vector{Float64},
#                         x::Vector{Float64})
#
# Computes and returns linear regression objective (mean squared error).
function LinearRegressionObj(A::Matrix{Float64}, b::Vector{Float64},
                             x::Vector{Float64})
  margins = A * x - b;
  return mean(margins.^2) / 2;
end

# x = ProximalUpdateLinreg(a::Vector{Float64}, b::Float64,
#                              x_init::Vector, stepsize::Float64)
#
# Sets x to minimize
#
#  .5 * (a' * x - b)^2 + norm(x - x_init)^2 / (2 * stepsize)
#
# Returns the minimizing x.
function ProximalUpdateLinreg(a::Vector{Float64}, b::Float64,
                              x_init::Vector, stepsize::Float64)
  # Replace this code to return the correct update.
  return x_init;
end

# x = SGUpdateLinreg(a::Vector{Float64}, b::Float64,
#                    x_init::Vector, stepsize::Float64)
#
# Sets x to minimize the linear approximation to the standard squared
# loss for linear regression, or, if g is the gradient of the loss
#
#   F(x; (a, b)) = .5 * (a' * x - b)^2
#
# at the point x_init, updates
#
#   x = x_init - stepsize * g;
function SGUpdateLinreg(a::Vector{Float64}, b::Float64,
                        x_init::Vector, stepsize::Float64)
  # Replace this code to return the correct update.
  return x_init;
end

# x = TruncatedUpdateLinreg(a::Vector{Float64}, b::Float64,
#                              x_init::Vector, stepsize::Float64)
#
# Let F(x; (a, b)) = .5 * (a' * x - b)^2. Sets x to minimize the
# positive approximation to F at x_init, that is, for
#
#  F_lin(x) = (F(x_init) + F'(x_init) * (x - x_init))_+
#
# sets x to minimize
#
#  F_lin + norm(x - x_init)^2 / (2 * stepsize).
#
# Returns the minimizing x.
function TruncatedUpdateLinreg(a::Vector{Float64}, b::Float64,
                               x_init::Vector, stepsize::Float64)
  # Replace this code to return the correct update.
  return x_init;
end
