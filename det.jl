using Distributions
using Statistics

struct Deterministic{T} <: Distributions.DiscreteUnivariateDistribution
    value::T
end

# Mean of the deterministic distribution
Statistics.mean(d::Deterministic) = d.value

# Variance of a deterministic distribution is zero
Statistics.var(d::Deterministic) = 0.0

# The deterministic distribution always returns its fixed value for sampling
Distributions.rand(d::Deterministic) = d.value

# Probability density or mass function
function Distributions.pdf(d::Deterministic, x)
    x == d.value ? 1.0 : 0.0
end

# Cumulative distribution function
function Distributions.cdf(d::Deterministic, x)
    x < d.value ? 0.0 : 1.0
end

# Support of the distribution
Distributions.support(d::Deterministic) = [d.value]

Base.iterate(d::Deterministic, state=nothing) = state === nothing ? (d.value, true) : nothing

# Pretty-printing
Base.show(io::IO, d::Deterministic) = print(io, "Deterministic(value=$(d.value))")

# Quantile method (if required for compatibility)
function Distributions.quantile(d::Deterministic, p::Real)
    return d.value
end