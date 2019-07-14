# phase-retrieval.jl
#
# Code for phase retrieval problems


# (A, b, x) = GeneratePhaseRetrievalData(m::Int64 = 200, n::Int64 = 20;
#                                        condition_number::Float64 = 1.0)
#
# Generates an m-by-n matrix A of measurement vectors a_i = A[i, :], where each
# a_i is uniform on the unit sphere. Chooses x ~ N(0, I), and then sets each
# observation b_i = (a_i' * x)^2.
function GeneratePhaseRetrievalData(m::Int64 = 200, n::Int64 = 20;
                                    condition_number::Float64 = 1.0)
  A = randn(m, n);
  A = A ./ sqrt.(repeat(sum(A.^2, dims=2), outer = (1, n)));  
  if (condition_number > 1.0)
    # Make A have average norm 1 columns
    condition_vec = Vector(LinRange(1.0, condition_number, n));
    condition_vec = condition_vec / norm(condition_vec);
    A = A .* repeat(condition_vec', outer = (m, 1));
  end
  x = randn(n);
  b = (A * x).^2;
  return (A, b, x);
end

# PhaseRetrievalObj(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64})
#
# Computes the (average) absolute loss for the phase retrieval problem
#
#  p(b | lambda) = exp(-lambda) * lambda^b / b!
#
# For a data vector a, the parameter lambda = exp(a' * x), so we have log loss
# (ignoring the factorial term)
#
# l(x; (a, b)) = -b * dot(a, x) + exp(dot(a, x)).
function PhaseRetrievalObj(A::Matrix{Float64}, b::Vector{Float64},
                           x::Vector{Float64})
  mm = size(A, 1);
  objective = sum(abs.((A * x).^2 - b)) / mm;
  return objective;
end

# x = ProximalUpdatePhaseRetrieval(a::Vector{Float64}, b::Float64,
#                              x_init::Vector, stepsize::Float64; accuracy)
#
# Assuming b >= 0, sets x to be the minimizer of
#
#   |(a'*x)^2 - b| + ||x - x_init||^2 / (2 * alpha).
function ProximalUpdatePhaseRetrieval(a::Vector{Float64}, b::Float64,
                                      x_init::Vector{Float64},
                                      stepsize::Float64)
  # Replace this code to return the correct update.
  return x_init;
end

# x = TruncatedUpdatePhaseRetrieval(a, b, x_init, stepsize)
#
# Let F(x, a, b) = abs((a'*x)^2 - b) and
#
#  F_approx(x) = max{ F(x_init) + F'(x_init) * (x - x_init)), 0}
#                     
# Sets x to minimize
#
#  F_approx(x) + norm(x - x_init)^2 / (2 * stepsize).
#
# Returns the minimizing x.
function TruncatedUpdatePhaseRetrieval(a::Vector{Float64}, b::Float64,
                                       x_init::Vector{Float64},
                                       stepsize::Float64)
  # Replace this code to return the correct update.
  return x_init;
end

# x = SGUpdatePhaseRetrieval(a::Vector{Float64}, b::Float64,
#                     x_init::Vector{Float64}, stepsize::Float64)
#
# Computes the stochastic gradient update for PhaseRetrieval regression.
function SGUpdatePhaseRetrieval(a::Vector{Float64}, b::Float64,
                                x_init::Vector{Float64}, stepsize::Float64)
  # Replace this code to return the correct update.
  return x_init;
end
