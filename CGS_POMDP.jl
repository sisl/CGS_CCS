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
    collected_locs::Vector{Geometry}
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

function observe(pomdp::CGSPOMDP, point::Point, layer::Int, column::Symbol)
    f = pomdp.belief[layer][column]
    x = pcu(point)
    y = pomdp.state.earth[layer].gt[point, column]
    p_fx = posterior(f(x, 0.1), y)
    pomdp.belief[layer][column] = p_fx

    return p_fx(x, 0.1)
end

function POMDPs.actions(pomdp::CGSPOMDP)::Vector{NamedTuple{(:id, :geometry), Tuple{Symbol, Geometry}}}
    well_actions = [(id=:well_action, geometry=x) for x in pomdp.state.well_logs]
    seismic_actions = [(id=:seismic_action, geometry=x) for x in pomdp.state.seismic_lines]
    observe_actions = [(id=:observe_action, geometry=x) for x in domain(pomdp.state.earth[1].gt)]
    return [well_actions; seismic_actions; observe_actions]
end

function observe(pomdp::CGSPOMDP, geom::Point)
    return vcat([[observe(pomdp, geom, layer, feature) for feature in pomdp.feature_names] for layer in 1:NUM_LAYERS]...)
end

function get_cond_distr(pomdp::CGSPOMDP, pts)::Vector{MvNormal}
    cond_distributions::Vector{MvNormal} = []
    grid = CartesianGrid(100, 100)

    myl = ReentrantLock()
    @threads for layer in 1:NUM_LAYERS
        local cond_distributions_layer = []
        
        gtlayer = pomdp.state.earth[layer].gt
        data_at_all_wells = gtlayer[Multi([pomdp.collected_locs...]), :]
        
        for column in pomdp.feature_names
            proc = GaussianProcess(
                SphericalVariogram(range=RANGE, sill=SILL, nugget=NUGGET),
                mean(gtlayer[:, column])
            ) # Switch to Abstract GPs.jl
            
            tgt_geom = Multi([pts...])
            tgt_realizations = [rand(proc, grid, data_at_all_wells)[tgt_geom, :] for _ in 1:(3 * length(pts))]
            
            Z = [tgt_rlz[:, column] for tgt_rlz in tgt_realizations]
            Z_matrix = hcat(Z...)
            mean_vector = mean(Z_matrix, dims=2) |> vec
            cov_matrix = cov(Z_matrix') + 1e-5 * I
            
            feat_distr = MvNormal(mean_vector, cov_matrix)
            push!(cond_distributions_layer, feat_distr)
        end

        @lock myl append!(cond_distributions, cond_distributions_layer)
    end
    return cond_distributions
end

function Base.:*(p::Point, scalar::Number)
    Point(p.coords.x * scalar, p.coords.y * scalar)
end

function Base.:+(p1::Point, p2::Point)
    Point(p1.coords.x + p2.coords.x, p1.coords.y + p2.coords.y)
end

function observe(pomdp::CGSPOMDP, geom::Segment)
    p1 = geom.vertices[1]
    p2 = geom.vertices[2]
    points = [p1 * (1 - t) + p2 * t for t in range(0, stop=1, length=20)]
    # TODO: Distribute this work
    return get_cond_distr(pomdp, points)
end

function POMDPs.observation(pomdp::CGSPOMDP, action, state) # returns distribution
    if action.id == :well_action # well log
        push!(pomdp.collected_locs, action.geometry)
        return product_distribution(observe(pomdp, action.geometry))
    elseif action.id == :observe_action
        return product_distribution(observe(pomdp, action.geometry))
    elseif action.id == :seismic_action
        push!(pomdp.collected_locs, action.geometry)
        return product_distribution(observe(pomdp, action.geometry))
    end
    return # product distribution of all things (up to 150 things, is this an issue)
end

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

function calculate_map_uncertainty_suitability(pomdp::CGSPOMDP)
    total_suitability_score::Float64 = 0.0
    layer_col_unc::Float64 = 0.0
    
    γ = SphericalVariogram(range=RANGE, sill=SILL, nugget=NUGGET) # Each feature can have a different nugget in the future.
    okrig = GeoStatsModels.OrdinaryKriging(γ)

    for layer in 1:NUM_LAYERS
        gtlayer = pomdp.state.earth[layer].gt
        data_at_all_wells = gtlayer[Multi([pomdp.collected_locs...]), :]
        fitkrig = GeoStatsModels.fit(okrig, data_at_all_wells)

        for pt in domain(pomdp.state.earth[1].gt)
            # If I am a point with high confidence and a high/low suitability score, I update map_suitability
            high_confidence::Bool = true # Do more math here
            suitability_score = 0.0
            for column in pomdp.feature_names
                layer_col_unc += GeoStatsModels.predictprob(fitkrig, column, pt).σ
                feat_distr = GeoStatsModels.predictprob(fitkrig, column, pt)
                if feat_distr.σ > HIGH_CONFIDENCE_THRESHOLD # TODO Make hyperparameter
                    high_confidence = false
                    break
                end
                suitability_score += score_component(column, feat_distr.μ)
            end
            if high_confidence
                total_suitability_score += (suitability_score > SUITABILITY_THRESHOLD) ? 1 : SUITABILITY_BIAS
            end
        end
    end
    return total_suitability_score
end

function calculate_map_suitability(pomdp::CGSPOMDP)
    total_suitability_score::Float64 = 0.0
    for layer in 1:NUM_LAYERS
        gtlayer = pomdp.state.earth[layer].gt
        data_at_all_wells = gtlayer[Multi([pomdp.collected_locs...]), :]
        γ = SphericalVariogram(range=RANGE, sill=SILL, nugget=NUGGET) # Each feature can have a different nugget in the future.
        
        okrig = GeoStatsModels.OrdinaryKriging(γ)
        fitkrig = GeoStatsModels.fit(okrig, data_at_all_wells)
        
        for pt in domain(pomdp.state.earth[1].gt)
            # If I am a point with high confidence and a high/low suitability score, I update map_suitability
            high_confidence::Bool = true # Do more math here (be more rigorous via sampling)
            suitability_score = 0.0
            for column in pomdp.feature_names
                feat_distr = GeoStatsModels.predictprob(fitkrig, column, pt)
                if feat_distr.σ > 1 # TODO Make hyperparameter
                    high_confidence = false
                    break
                end
                suitability_score += score_component(column, feat_distr.μ)
            end
            if high_confidence
                total_suitability_score += (suitability_score > SUITABILITY_THRESHOLD) ? 1 : SUITABILITY_BIAS
            end
        end
    end
    return total_suitability_score
end

function calculate_map_uncertainty(pomdp::CGSPOMDP)::Float64
    layer_col_unc::Float64 = 0.0
    for column in pomdp.feature_names
        for layer in 1:NUM_LAYERS
            gtlayer = pomdp.state.earth[layer].gt
            data_at_all_wells = gtlayer[Multi([pomdp.collected_locs...]), :]

            γ = SphericalVariogram(range=RANGE, sill=SILL, nugget=NUGGET) # Each feature can have a different nugget in the future.
            okrig = GeoStatsModels.OrdinaryKriging(γ)
            fitkrig = GeoStatsModels.fit(okrig, data_at_all_wells)

            for pt in domain(pomdp.state.earth[1].gt)
                layer_col_unc += GeoStatsModels.predictprob(fitkrig, column, pt).σ
            end
        end
    end
    pomdp.map_uncertainty = layer_col_unc
    return layer_col_unc
end

function reward_information_suitability(pomdp::CGSPOMDP)
    r_unc::Float64 = pomdp.map_uncertainty
    r_suit::Float64 = calculate_map_uncertainty_suitability(pomdp)
    if r_unc < 0
        r_unc = 1.0
    else
        r_unc -= pomdp.map_uncertainty
    end
    return r_unc + r_suit
end

function reward_information_gain(pomdp::CGSPOMDP)
    r::Float64 = pomdp.map_uncertainty
    if r < 0
        calculate_map_uncertainty(pomdp)
        return 1.0 # If its the first action, we don't know how much better we've done
    end
    r -= calculate_map_uncertainty(pomdp)
    return r
end


function POMDPs.reward(pomdp::CGSPOMDP, state, action)
    return reward_action_cost(action) + reward_information_suitability(pomdp)
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