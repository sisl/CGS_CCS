#%%
using POMDPs
using Distributions
using GeoStats
using DataFrames
import GLMakie as Mke
using Infiltrator

# CONSTANTS
const NUM_LAYERS = 5
const NUM_LINES = 10
const RANGE = 15.
const SILL = 3.0
const NUGGET = .1

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

struct SeismicLine
    x1::Number
    x2::Number
    y1::Number
    y2::Number
end

"""
A type for the state of the CGS POMDP
Currently, we allow a Vector of LayerFeatures to simulate a layer cake model
and a vector of seismic lines
"""
struct CGS_State
    earth::Vector{LayerFeatures}
    seismic_lines::Vector{SeismicLine}
end

struct CGSPOMDP <: POMDP{CGS_State, Symbol, Symbol}
    state::CGS_State
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
        lines = [SeismicLine(rand(0.0:100.0), rand(0.0:100.0), rand(0.0:100.0), rand(0.0:100.0))
                    for _ in 0:NUM_LINES]
        return new(CGS_State(earth, lines))
    end
end

pomdp = CGSPOMDP()

# function POMDPs.states(pomdp::CGSPOMDP)
#     return [pomdp.state]
# end

# function POMDPs.transition(pomdp::CGSPOMDP, state, action) # actually have a state transition???
#     return SparseCat([state], [1.0])
# end


# ----------------- Taken from tiger -----------------
# # Define the actions: open left, open right, or listen
# function POMDPs.actions(pomdp::TigerPOMDP)
#     return [:open_left, :open_right, :listen]
# end

# # Define the observations: hear tiger on the left, hear tiger on the right
# function POMDPs.observations(pomdp::TigerPOMDP)
#     return [:hear_left, :hear_right]
# end


# # Observation model
# function POMDPs.observation(pomdp::TigerPOMDP, action, state, next_state)
#     if action == :listen
#         if next_state
#             return SparseCat([:hear_left, :hear_right], [0.85, 0.15]) # Tiger on the left
#         else
#             return SparseCat([:hear_left, :hear_right], [0.15, 0.85]) # Tiger on the right
#         end
#     else
#         return SparseCat([:hear_left, :hear_right], [0.5, 0.5]) # Random observation after opening a door
#     end
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
