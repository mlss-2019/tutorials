# logreg.jl
#
# Code for logistic regression, but this version uses mini-batches.
#
# In this case, for a triple (a, b, x), where a, x are n-vectors and b
# is {-/+ 1} valued, the objective is
#
#  F(x, a, b) = log(1 + exp(-b * a' * x)).

using Distributions;

# (A, b, x) = GenerateLogisticData(m::Int64 = 200, n::Int64 = 20;
#                                  condition_number::Float64 = 1.0,
#                                  snr::Float64 = 1.0)
#
# Generates logistic regression data where the data matrix A is m-by-n,
# the label (target) vector b is length m, and the distribution of b is
#
#  p(b | a, x) = exp(b * a' * x) / (1 + exp(b * a' * x)),
#
# for b in {-1, 1}. The parameter snr governs the mean magnitude of
# the inner products A * x, that is, the margins. (High snr means less
# noise in the labels).
function GenerateLogisticData(m::Int64 = 200, n::Int64 = 20;
                              condition_number::Float64 = 1.0,
                              snr::Float64 = 1.0)
  A = randn(m, n);
  if (condition_number > 1.0)
    # Make A have condition number given
    condition_vec = Vector(LinRange(1.0, condition_number, n));
    condition_vec = condition_vec / norm(condition_vec);
    A = A .* repeat(condition_vec', outer = (m, 1));
  end
  x = randn(n);
  mean_inner_product = mean(abs.(A * x));
  # Make it so that the mean inner product between data and x is equal
  # to snr (the signal to noise ratio).
  x = snr * x / mean_inner_product;
  margins = A * x;
  probs = 1 ./ (1 .+ exp.(-margins));
  b = sign.(rand(m) .- 1 + probs);
  return (A, b, x);
end

# o = LogisticRegressionObj(A::Matrix{Float64}, b::Vector{Float64},
#                           x::Vector{Float64})
#
# Computes and returns logistic regression objective (mean log loss).
function LogisticRegressionObj(A::Matrix{Float64}, b::Vector{Float64},
                               x::Vector{Float64})
  margins = b .* (A * x);
  pos_inds = (margins .> 0);
  neg_inds = (margins .<= 0);
  obj = (sum(log.(1 .+ exp.(-margins[pos_inds])))
         + sum(log.(1 .+ exp.(margins[neg_inds])) - margins[neg_inds]));
  return obj / size(A, 1);
end

# g = LogisticRegressionGrad(A::Matrix{Float64}, b::Vector{Float64},
#                            x::Vector{Float64})
#
# Computes and returns the gradient of the logistic regression loss.
function LogisticRegressionGrad(A::Matrix{Float64}, b::Vector{Float64},
                                x::Vector{Float64})
  margins = b .* (A * x);
  probs = b ./ (1 .+ exp.(margins));
  grad = -A' * probs;
  return grad / size(A, 1);
end

# x = SGUpdateLogistic(A::SubArray{Float64, 2}, b::SubArray{Float64, 1},
#                        x_init::Vector{Float64}, stepsize::Float64)
#
# Computes and applies a minibatch stochastic gradient update to the
# vector x_init for the logistic regression loss. The entire minibatch is
# in the SubArrays (A, b), which can be treated like matrices.
function SGUpdateLogistic(A::SubArray{Float64, 2}, b::SubArray{Float64, 1},
                          x_init::Vector{Float64}, stepsize::Float64)
  # Replace this code to return the correct update.
  return x_init;
end

# x = TruncatedUpdateLogistic(A::SubArray{Float64, 2},
#                             b::SubArray{Float64, 1},
#                             x_init::Vector{Float64}, stepsize::Float64)
#
# Performs a mini-batch update using the truncated model for logistic
# regression. In particular, letting a_i be the rows of the matrix A and
# b_i the associated labels, uses that the logistic loss has lower-bound 0
# to make the approximation
#
#  F_approx(x) = max{ F_0 + g' * (x - x_init), 0 }
#
# where F_0 = (1/m) sum_{i=1}^m log(1 + exp(-b_i * a_i' * x)) and
#
#   g = (d / dx) (1/m) sum_{i = 1}^m log(1 + exp(-b_i * a_i' * x))
#
# is the gradient on the mini-batch.
function TruncatedUpdateLogistic(A::SubArray{Float64, 2},
                                 b::SubArray{Float64, 1},
                                 x_init::Vector{Float64}, stepsize::Float64)
  # Replace this code to return the correct update.
  return x_init;
end

function TruncatedUpdateLogistic(a::Vector{Float64},
                                 b::Float64,
                                 x_init::Vector{Float64}, stepsize::Float64)
  # Replace this code to return the correct update.
  return x_init;
end

function SGUpdateLogistic(a::Vector{Float64}, b::Float64,
                          x_init::Vector{Float64}, stepsize::Float64)
  # Replace this code to return the correct update.
  return x_init;
end
