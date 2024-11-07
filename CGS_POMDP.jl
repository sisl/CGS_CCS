#%%
using POMDPs
using Distributions
using GeoStats
using DataFrames
using LinearAlgebra
import GLMakie as Mke
using Infiltrator
using Base.Threads
using AbstractGPs
using Unitful

include("config.jl")

struct GeoFeatures
    range::Number
    sill::Number
    nugget::Number
    name::Symbol
    mean::Number
end
# TODO: Add variable uncertainty!
"""
LayerFeatures struct

Holds a GeoTable for each Layer, with measurements for each feature
use viewer(gt) to visualize a Layer
"""
struct LayerFeatures
    gt::GeoTable
    df::DataFrame
    function LayerFeatures(feats::Vector{GeoFeatures})
        grid = CartesianGrid(100, 100)
        dfs::Vector{DataFrame} = []
        for feat in feats
            proc = GaussianProcess(SphericalVariogram(range=feat.range, sill=feat.sill, nugget=feat.nugget), feat.mean)
            simul = rand(proc, grid, [feat.name => Float64], 1)[1]
            feat_dataframe = DataFrame(simul)
            select!(feat_dataframe, Not(:geometry))
            push!(dfs, feat_dataframe)
        end
        all_feature_df = hcat(dfs...)
        all_feature_gt = georef(all_feature_df, grid)
        
        return new(all_feature_gt, all_feature_df)
    end
end


"""
A type for the state of the CGS POMDP
Currently, we allow a Vector of LayerFeatures to simulate a layer cake model
and a vector of seismic lines
"""
struct CGS_State
    earth::Vector{LayerFeatures}
    seismic_lines::Vector{Segment}
    well_logs::Vector{Point}
end

mutable struct CGSPOMDP <: POMDP{CGS_State, Symbol, Symbol}
    state::CGS_State
    feature_names::Vector{Symbol}
    map_uncertainty::Float64 # Some measure of uncertainty over the whole map
    belief::Vector{Dict{Symbol, Any}} # Every layer/feature combo is an independent GP
    function initialize_earth()::Vector{LayerFeatures}
        randlayers::Vector{LayerFeatures} = []
        prev_mean = 0.
        for layer in 1:NUM_LAYERS
            layer_params::Vector{GeoFeatures} = []
            # For z
            if layer == 1
                push!(layer_params, GeoFeatures(RANGE, SILL, NUGGET, :z, 0.))
            else
                prev_mean += rand(3:11)
                push!(layer_params, GeoFeatures(RANGE, SILL, NUGGET, :z, prev_mean))
            end
            
            # For permeability
            λ = 100
            push!(layer_params, GeoFeatures(RANGE, SILL, NUGGET, :permeability, rand(Exponential(λ))))
            
            # For top seal thickness
            push!(layer_params, GeoFeatures(RANGE, SILL, NUGGET, :topSealThickness, rand(10:80)))

            push!(randlayers, LayerFeatures(layer_params))
        end
        return randlayers
    end
    function CGSPOMDP()
        earth = initialize_earth()
        feature_names = [:z, :permeability, :topSealThickness]
        lines = [Segment(Point(rand(0.0:100.0), rand(0.0:100.0)), Point(rand(0.0:100.0), rand(0.0:100.0)))
                    for _ in 1:NUM_LINES] # TODO: Make lines more realistic (longer)
        wells = [Point(rand(0.0:100.0), rand(0.0:100.0)) for _ in 1:NUM_WELLS]
        gps = [Dict(
                    :z => GP(Matern32Kernel()), 
                    :permeability => GP(Matern32Kernel()),
                    :topSealThickness => GP(Matern32Kernel())
                ) for _ in 1:NUM_LAYERS]
        return new(CGS_State(earth, lines, wells), [], feature_names, -1.0, gps)
    end
end

function POMDPs.states(pomdp::CGSPOMDP)
    return [pomdp.state]
end

function POMDPs.transition(pomdp::CGSPOMDP, state, action)
    return SparseCat([state], [1.0])
end

function buy_well_data(pomdp::CGSPOMDP, ind::Int)
    well = pomdp.state.well_logs[ind]
    push!(pomdp.collected_locs, well)
end



function POMDPs.actions(pomdp::CGSPOMDP)::Vector{NamedTuple{(:id, :geometry), Tuple{Symbol, Geometry}}}
    well_actions = [(id=:well_action, geometry=x) for x in pomdp.state.well_logs]
    seismic_actions = [(id=:seismic_action, geometry=x) for x in pomdp.state.seismic_lines]
    observe_actions = [(id=:observe_action, geometry=x) for x in domain(pomdp.state.earth[1].gt)]
    return [well_actions; seismic_actions; observe_actions]
end

function observe(pomdp::CGSPOMDP, point::Point, layer::Int, column::Symbol)
    f = pomdp.belief[layer][column]
    x = pcu(point)
    y = pomdp.state.earth[layer].gt[point, column]
    p_fx = posterior(f(x, 0.1), y)
    pomdp.belief[layer][column] = p_fx

    mean_cond = mean(p_fx)
    cov_cond = cov(p_fx)
    joint_conditional_dist = MvNormal(mean_cond, cov_cond)
    return joint_conditional_dist
end

function observe(pomdp::CGSPOMDP, geom::Segment, layer::Int, column::Symbol)
    p1 = geom.vertices[1] # if slow we can move these lines to the calling fn
    p2 = geom.vertices[2]
    points = [p1 * (1 - t) + p2 * t for t in range(0, stop=1, length=20)] 
    
    f = pomdp.belief[layer][column]
    x = pcu(points)
    y = pomdp.state.earth[layer].gt[points, column]
    p_fx = posterior(f(x, 0.1), y)
    pomdp.belief[layer][column] = p_fx

    mean_cond = mean(p_fx)
    cov_cond = cov(p_fx)
    joint_conditional_dist = MvNormal(mean_cond, cov_cond)
    return joint_conditional_dist
end

function observe(pomdp::CGSPOMDP, geom)
    return vcat([[observe(pomdp, geom, layer, feature) for feature in pomdp.feature_names] for layer in 1:NUM_LAYERS]...)
end

function Base.:*(p::Point, scalar::Number)
    Point(p.coords.x * scalar, p.coords.y * scalar)
end

function Base.:+(p1::Point, p2::Point)
    Point(p1.coords.x + p2.coords.x, p1.coords.y + p2.coords.y)
end


POMDPs.observation(pomdp::CGSPOMDP, action, state) = return product_distribution(observe(pomdp, action.geometry))


function reward_action_cost(action::NamedTuple{(:id, :geometry), Tuple{Symbol, Geometry}})
    if action.id == :well_action
        return WELL_COST
    elseif action.id == :seismic_action
        return SEISMIC_LINE_COST
    elseif action.id == :observe_action
        return 0
    else
        return 0
    end
end

function score_component(feature::Symbol, value)
    if feature == :permeability
        if value < 10
            return 0
        elseif value < 20
            return 1
        elseif value < 50
            return 2
        elseif value < 100
            return 4
        elseif value < 500
            return 5
        else
            return 3
        end
    elseif feature == :topSealThickness
        value = (value - 25) ÷ 10
        value = clamp(value, 0, 5)
        return value
    elseif feature == :z
        if value < 800
            return 0
        elseif value < 1000
            return 1
        elseif value < 1500
            return 3
        elseif value < 2000
            return 5
        elseif value < 3000
            return 4
        else
            return 2
        end
    end
end

function calculate_map_uncertainty(pomdp::CGSPOMDP)
    layer_col_unc::Float64 = 0.0

    for layer in 1:NUM_LAYERS
        for column in pomdp.feature_names
            gridx = pcu([pt.vertices[1] for pt in domain(pomdp.state.earth[1].gt)])
            fgrid = pomdp.belief[layer][column](gridx)
            fs = marginals(fgrid)
            marginal_stds = std.(fs)
            layer_col_unc += sum(marginal_stds)
        end
    end
    pomdp.map_uncertainty = layer_col_unc
    return layer_col_unc
end

function reward_information_gain(pomdp::CGSPOMDP)
    r::Float64 = pomdp.map_uncertainty
    if r < 0
        calculate_map_uncertainty(pomdp)
        return 1.0 # If its the first action, we don't know how much better we've done
        # Does this incentivize bad first actions?? Most likely yes, talk to Mansur
    end
    r -= calculate_map_uncertainty(pomdp)
    return r
end

function reward_suitability(pomdp::CGSPOMDP)
    total_grid_suitability = 0.
    for layer in 1:NUM_LAYERS
        gridx = pcu([pt.vertices[1] for pt in domain(pomdp.state.earth[1].gt)])
        for pt in gridx
            # For each column gerate a distribution
            pt_score_samples = zeros(SUITABILITY_NSAMPLES)
            for column in pomdp.feature_names
                fgrid = pomdp.belief[layer][column]([pt])
                pt_score_samples .+= [score_component(column, rand(fgrid)) for _ in 1:SUITABILITY_NSAMPLES]
            end
            pt_score_samples ./= length(pomdp.feature_names)
            cnt_better = mean(pt_score_samples .> 3.5)
            if cnt_better > 0.8
                total_grid_suitability += 1.0
            end
            cnt_worse = mean(pt_score_samples .< 3.5)
            if cnt_worse > 0.8
                total_grid_suitability += SUITABILITY_BIAS
            end
        end
    end
    return total_grid_suitability
end

function POMDPs.reward(pomdp::CGSPOMDP, state, action)
    return reward_action_cost(action) + reward_information(pomdp) + reward_suitability(pomdp)
end

POMDPs.discount(pomdp::CGSPOMDP) = 0.95 

# Utility fns
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