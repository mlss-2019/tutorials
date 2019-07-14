# utilities.jl
#
# A few simple bindings because I am lazy.

function linspace(a::Real, b::Real, n::Int64)
  return LinRange(a, b, n);
end

function logspace(a::Real, b::Real, n::Int64)
  return 10.0.^LinRange(a, b, n);
end
