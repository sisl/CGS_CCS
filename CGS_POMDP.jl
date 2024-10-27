#%%
using POMDPs
using Distributions
using GeoStats
using DataFrames
using LinearAlgebra
import GLMakie as Mke
using Infiltrator

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
    beliefs::Dict
    function LayerFeatures(feats::Vector{GeoFeatures})
        gps = Dict()
        grid = CartesianGrid(100, 100)
        dfs::Vector{DataFrame} = []
        for feat in feats
            proc = GaussianProcess(SphericalVariogram(range=feat.range, sill=feat.sill, nugget=feat.nugget), feat.mean)
            gps[feat.name] = proc
            simul = rand(proc, grid, [feat.name => Float64], 1)[1]
            feat_dataframe = DataFrame(simul)
            select!(feat_dataframe, Not(:geometry))
            push!(dfs, feat_dataframe)
        end
        all_feature_df = hcat(dfs...)
        all_feature_gt = georef(all_feature_df, grid)
        
        return new(all_feature_gt, all_feature_df, gps)
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
        return new(CGS_State(earth, lines, wells), [], feature_names)
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
    gtlayer = pomdp.state.earth[layer].gt
    data_at_all_wells = gtlayer[Multi([pomdp.collected_locs...]), :]

    γ = SphericalVariogram(range=RANGE, sill=SILL, nugget=NUGGET) # Each feature can have a different nugget in the future.
    okrig = GeoStatsModels.OrdinaryKriging(γ)
    fitkrig = GeoStatsModels.fit(okrig, data_at_all_wells)
    return GeoStatsModels.predictprob(fitkrig, column, point)
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

function get_cond_distr(layer::Int, column::Symbol, pts)
    gtlayer = pomdp.state.earth[layer].gt
    data_at_all_wells = gtlayer[Multi([pomdp.collected_locs...]), :]

    proc = GaussianProcess(
            SphericalVariogram(range=RANGE, sill=SILL, nugget=NUGGET), 
            mean(gtlayer[:, column])
            ) # TODO: This is a bit of a hack, we assume we have an okay estimate of mean for a layer
            
    grid = CartesianGrid(100, 100)

    tgt_geom = Multi([pts...])
    tgt_realizations = [rand(proc, grid, data_at_all_wells)[tgt_geom, :] for _ in 1:100];

    Z = [tgt_rlz[:, column] for tgt_rlz in tgt_realizations]
    Z_matrix = hcat(@view Z...)
    mean_vector = mean(Z_matrix, dims=2) |> vec
    # println(size(Z_matrix))
    cov_matrix = cov(Z_matrix') + 1e-5 * I
    # println(eigen(cov_matrix).values)
    # @assert all(eigen(cov_matrix).values .> 0)
    feat_distr = MvNormal(mean_vector, cov_matrix)
    return feat_distr
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
    return vcat([[get_cond_distr(layer, feature, points) for feature in pomdp.feature_names] for layer in 1:NUM_LAYERS]...)
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

# ----------------- Taken from tiger -----------------
# # Define the actions: open left, open right, or listen


# # Define the observations: hear tiger on the left, hear tiger on the right
# function POMDPs.observations(pomdp::TigerPOMDP)
#     return [:hear_left, :hear_right]
# end




# # Reward model
# function POMDPs.reward(pomdp::TigerPOMDP, state, action)
#     if action == :open_left
#         return state ? -100.0 : 10.0  # Penalty for opening the door with the tiger, reward for the empty one
#     elseif action == :open_right
#         return state ? 10.0 : -100.0
#     else
#         return -1.0  # Cost of listening
#     end
# end

# # Discount factor
# POMDPs.discount(pomdp::TigerPOMDP) = pomdp.discount

# # Create the POMDP object
# pomdp = TigerPOMDP(0.95)

# # Solve the POMDP using a value iteration solver
# solver = ValueIterationSolver()
# policy = solve(solver, pomdp)

# println("Optimal policy computed!")
