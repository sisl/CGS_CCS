"""
Deterministic distribution for initialstate and transition models
"""
struct Deterministic{T} <: Distributions.DiscreteUnivariateDistribution
    value::T
end

Statistics.mean(d::Deterministic) = d.value
Statistics.var(d::Deterministic) = 0.0
Distributions.rand(d::Deterministic) = d.value

Distributions.pdf(d::Deterministic, x) = x == d.value ? 1.0 : 0.0
Distributions.cdf(d::Deterministic, x) = x < d.value ? 0.0 : 1.0

Distributions.support(d::Deterministic) = [d.value]

Base.iterate(d::Deterministic, state=nothing) = state === nothing ? (d.value, true) : nothing
Base.show(io::IO, d::Deterministic) = print(io, "Deterministic(value=$(d.value))")

Distributions.quantile(d::Deterministic, p::Real) = d.value # Not sure about this

# Geostats to vector utility fns
"""
Point Conversion Utility (pcu)
Take a 2D Point from GeoStats.jl and convert it into a 
1 element vector of vector of 2 coordinates that is easily fed into a GP from
AbstractGPs.jl
E.g.
>>> point = Point(1.0, 3.0)
>>> pcu(point)
1-element Vector{Vector{Quantity{Float64, 𝐋, Unitful.FreeUnits{(m,), 𝐋, nothing}}}}:
[1.0, 3.0] # Need to strip units as well for AbstractGPs
"""
function pcu(p::Point)
    return [[ustrip(p.coords.x), ustrip(p.coords.y)]]
end

function pcu(pts::Vector{<:Point})
    return [[ustrip(p.coords.x), ustrip(p.coords.y)] for p in pts]
end


function Base.:*(p::Point, scalar::Number)
    Point(p.coords.x * scalar, p.coords.y * scalar)
end

function Base.:+(p1::Point, p2::Point)
    Point(p1.coords.x + p2.coords.x, p1.coords.y + p2.coords.y)
end

function dist(p1::Point, p2::Point)
    √((p1.coords.x - p2.coords.x) ^ 2 + (p1.coords.y - p2.coords.y) ^ 2)
end