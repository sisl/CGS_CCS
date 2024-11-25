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


# Visualization functions.
"""
Visualize the uncertainty in the belief of a layer and feature.
"""
function visualize_uncertainty(pomdp::CCSPOMDP, layer::Int, column::Symbol)
    var_mtx = zeros(GRID_SIZE, GRID_SIZE)
    gridx = pcu([pt.vertices[1] for pt in domain(pomdp.state.earth[layer].gt)]) 
    all_rock_mean = zeros(length(gridx))
    for rocktype in 1:length(instances(RockType))
        if pomdp.rocktype_belief[layer].p[rocktype] == 0.0
            continue
        end
        ms = marginals(pomdp.belief[rocktype][layer][column](gridx))
        all_rock_mean += mean.(ms) * pomdp.rocktype_belief[layer].p[rocktype]
    end

    # Eqn for mixture distribution: https://en.wikipedia.org/wiki/Mixture_distribution
    for rocktype in 1:length(instances(RockType))
        if pomdp.rocktype_belief[layer].p[rocktype] == 0.0
            continue
        end
        ms = marginals(pomdp.belief[rocktype][layer][column](gridx))
        mg_stds = std.(ms)
        mg_means = mean.(ms)
        var_compontent = ((mg_stds .^ 2) + (mg_means - all_rock_mean) .^ 2) * pomdp.rocktype_belief[layer].p[rocktype]
        var_mtx .+= reshape(var_compontent, GRID_SIZE, GRID_SIZE)'
    end

    xaxisvalues = yaxisvalues = (1:GRID_SIZE) .* SPACING
    heatmap(xaxisvalues, 
            yaxisvalues,
            sqrt.(var_mtx),
            color=:viridis,
            xlabel="Grid X (m)",
            ylabel="Grid Y (m)",
            title="Uncertainty Layer $layer Feature $column",
            # clims=(0, maximum(mg_stds))
            )
end

function visualize_uncertainty(pomdp::CCSPOMDP, layer::Int, column::Symbol, supplementary_points::Vector)
    var_mtx = zeros(GRID_SIZE, GRID_SIZE)
    gridx = pcu([[pt.vertices[1] for pt in domain(pomdp.state.earth[layer].gt)] ; supplementary_points]) 
    all_rock_mean = zeros(length(gridx))
    for rocktype in 1:length(instances(RockType))
        if pomdp.rocktype_belief[layer].p[rocktype] == 0.0
            continue
        end
        ms = marginals(pomdp.belief[rocktype][layer][column](gridx))
        all_rock_mean += mean.(ms) * pomdp.rocktype_belief[layer].p[rocktype]
    end

    # Eqn for mixture distribution: https://en.wikipedia.org/wiki/Mixture_distribution
    for rocktype in 1:length(instances(RockType))
        if pomdp.rocktype_belief[layer].p[rocktype] == 0.0
            continue
        end
        ms = marginals(pomdp.belief[rocktype][layer][column](gridx))
        mg_stds = std.(ms)
        println("Standard deviations of supplementary_points and 3 regular points: ", mg_stds[end - length(supplementary_points) - 3:end])
        mg_means = mean.(ms)
        var_compontent = ((mg_stds .^ 2) + (mg_means - all_rock_mean) .^ 2) * pomdp.rocktype_belief[layer].p[rocktype]
        var_mtx .+= reshape(var_compontent[1:end - length(supplementary_points)], GRID_SIZE, GRID_SIZE)'
    end

    xaxisvalues = yaxisvalues = (1:GRID_SIZE) .* SPACING
    heatmap(xaxisvalues, 
            yaxisvalues,
            sqrt.(var_mtx),
            color=:viridis,
            xlabel="Grid X (m)",
            ylabel="Grid Y (m)",
            title="Uncertainty Layer $layer Feature $column",
            # clims=(0, maximum(mg_stds))
            )
end

visualize_gt(pomdp::CCSPOMDP, layer::Int) = viewer(pomdp.state.earth[layer].gt)

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

function Base.:-(p1::Point, p2::Point) # Minus between points returns distance
    √((p1.coords.x - p2.coords.x) ^ 2 + (p1.coords.y - p2.coords.y) ^ 2)
end