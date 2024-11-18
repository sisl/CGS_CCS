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
        grid = CartesianGrid(GRID_SIZE, GRID_SIZE)
        dfs::Vector{DataFrame} = []
        for feat in feats
            proc = GaussianProcess(SphericalVariogram(range=feat.range, sill=feat.sill, nugget=feat.nugget), feat.mean)
            simul = GeoStats.rand(proc, grid, [feat.name => Float64], 1)[1]
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
A type for the state of the CCS POMDP
Currently, we allow a Vector of LayerFeatures to simulate a layer cake model
and a vector of seismic lines
"""
struct CCS_State
    earth::Vector{LayerFeatures}
    seismic_lines::Vector{Segment}
    well_logs::Vector{Point}
end

mutable struct CCSPOMDP <: POMDP{CCS_State, @NamedTuple{id::Symbol, geometry::Geometry}, Any}
    state::CCS_State
    feature_names::Vector{Symbol}
    map_uncertainty::Float64 # Some measure of uncertainty over the whole map
    belief::Vector{Dict{Symbol, Any}} # Every layer/feature combo is an independent GP
    action_index::Dict
    function initialize_earth()::Vector{LayerFeatures}
        randlayers::Vector{LayerFeatures} = []
        prev_mean = 0.
        for layer in 1:NUM_LAYERS
            layer_params::Vector{GeoFeatures} = []
            # For z
            if layer == 1
                push!(layer_params, GeoFeatures(RANGE, SILL, NUGGET, :z, 0.))
            else
                prev_mean += Base.rand(300:800)
                push!(layer_params, GeoFeatures(RANGE, SILL, NUGGET, :z, prev_mean))
            end
            
            # For permeability
            λ = 100
            push!(layer_params, GeoFeatures(RANGE, SILL, NUGGET, :permeability, Base.rand(Exponential(λ))))
            
            # For top seal thickness
            push!(layer_params, GeoFeatures(RANGE, SILL, NUGGET, :topSealThickness, Base.rand(10:80)))

            push!(randlayers, LayerFeatures(layer_params))
        end
        return randlayers
    end
    function initialize_belief(feature_names::Vector{Symbol})::Vector{Dict}
        starter_beliefs::Vector{Dict} = []
        for layer in 1:NUM_LAYERS
            layer_beliefs = Dict()
            for column in feature_names
                if column == :z
                    shift, scale = (500 * layer, 300 * 300)
                else
                    shift, scale = PRIOR_BELIEF[column]
                end
                layer_beliefs[column] = GP(shift, scale * Matern32Kernel())
            end
            push!(starter_beliefs, layer_beliefs)
        end
        return starter_beliefs
    end
    function CCSPOMDP()
        earth = initialize_earth()
        feature_names = [:z, :permeability, :topSealThickness]
        lines = [Segment(Point(Base.rand(0.0:float(GRID_SIZE)), 
                               Base.rand(0.0:float(GRID_SIZE))), 
                         Point(Base.rand(0.0:float(GRID_SIZE)),
                               Base.rand(0.0:float(GRID_SIZE))))
                    for _ in 1:NUM_LINES] # TODO: Make lines more realistic (longer)
        wells = [Point(Base.rand(0.0:float(GRID_SIZE)), 
                       Base.rand(0.0:float(GRID_SIZE))) for _ in 1:NUM_WELLS]
        gps = initialize_belief(feature_names)
        return new(CCS_State(earth, lines, wells), feature_names, -1.0, gps, Dict())
    end
end

include("utils.jl")

function POMDPs.states(pomdp::CCSPOMDP)
    return [pomdp.state]
end
function POMDPs.initialstate(pomdp::CCSPOMDP)
    return Deterministic(pomdp.state)
end
POMDPs.stateindex(pomdp::CCSPOMDP, state::CCS_State) = 1

POMDPs.transition(pomdp::CCSPOMDP, state, action) = Deterministic(pomdp.state)

function buy_well_data(pomdp::CCSPOMDP, ind::Int)
    well = pomdp.state.well_logs[ind]
    push!(pomdp.collected_locs, well)
end


function POMDPs.actions(pomdp::CCSPOMDP)::Vector{NamedTuple{(:id, :geometry), Tuple{Symbol, Geometry}}}
    well_actions = [(id=:well_action, geometry=x) for x in pomdp.state.well_logs]
    seismic_actions = [(id=:seismic_action, geometry=x) for x in pomdp.state.seismic_lines]
    observe_actions = [] # [(id=:observe_action, geometry=x.vertices[1]) for x in domain(pomdp.state.earth[1].gt)]
    return [well_actions; seismic_actions; observe_actions]
end

function POMDPs.actionindex(pomdp, action::@NamedTuple{id::Symbol, geometry::Geometry})
    if action in keys(pomdp.action_index)
        return pomdp.action_index[action]
    end
    all_actions = POMDPs.actions(pomdp)
    for (i, action) in enumerate(all_actions)
        pomdp.action_index[action] = i
    end
    return pomdp.action_index[action]
end

function observe(pomdp::CCSPOMDP, point::Point, layer::Int, column::Symbol, action_id::Symbol)
    f = pomdp.belief[layer][column]
    x = pcu(point)
    y = pomdp.state.earth[layer].gt[point, column]
    unc = ACTION_UNCERTAINTY[(action_id, column)]
    if unc < 0 # Feature belief not changed by action
        mean_cond = mean(f(x))
        cov_cond = cov(f(x))
    else
        p_fx = posterior(f(x, unc), y)
        pomdp.belief[layer][column] = p_fx
        
        mean_cond = mean(p_fx(x))
        cov_cond = cov(p_fx(x))
    end
    joint_conditional_dist = MvNormal(mean_cond, cov_cond)
    return joint_conditional_dist
end

function observe(pomdp::CCSPOMDP, geom::Segment, layer::Int, column::Symbol, action_id::Symbol)
    p1 = geom.vertices[1] # if slow we can move these lines to the calling fn
    p2 = geom.vertices[2]
    points = [p1 * (1 - t) + p2 * t for t in range(0, stop=1, length=SEISMIC_N_POINTS)] 
    
    f = pomdp.belief[layer][column]
    x = pcu(points)
    y = [pomdp.state.earth[layer].gt[pt, column][1] for pt in points]
    unc = ACTION_UNCERTAINTY[(action_id, column)]
    p_fx = posterior(f(x, unc), y)
    pomdp.belief[layer][column] = p_fx

    mean_cond = mean(p_fx(x))
    cov_cond = cov(p_fx(x))
    joint_conditional_dist = MvNormal(mean_cond, cov_cond)
    return joint_conditional_dist
end

function observe(pomdp::CCSPOMDP, action)
    return vcat([[observe(pomdp, 
                            action.geometry, 
                            layer, 
                            feature, 
                            action.id) 
                for feature in pomdp.feature_names] 
                for layer in 1:NUM_LAYERS]...)
end


POMDPs.observation(pomdp::CCSPOMDP, action, state) = return product_distribution(observe(pomdp, action))

function reward_action_cost(action::NamedTuple{(:id, :geometry), Tuple{Symbol, Geometry}})
    if action.id == :well_action
        return -WELL_COST
    elseif action.id == :seismic_action
        return -SEISMIC_LINE_COST
    elseif action.id == :observe_action
        return 0.
    else
        return 0.
    end
end

function score_component(feature::Symbol, value)
    if feature == :permeability
        if value < 10.
            return 0
        elseif value < 20.
            return 1
        elseif value < 50.
            return 2
        elseif value < 100.
            return 4
        elseif value < 500.
            return 5
        else
            return 3
        end
    elseif feature == :topSealThickness
        value = (value - 25) ÷ 10
        value = clamp(value, 0, 5)
        return value
    elseif feature == :z
        if value < 800.
            return 0
        elseif value < 1000.
            return 1
        elseif value < 1500.
            return 3
        elseif value < 2000.
            return 5
        elseif value < 3000.
            return 4
        else
            return 2
        end
    end
end

function calculate_map_uncertainty(pomdp::CCSPOMDP)
    layer_col_unc::Float64 = 0.0

    for layer in 1:NUM_LAYERS
        for column in pomdp.feature_names
            gridx = pcu([pt.vertices[1] for pt in domain(pomdp.state.earth[layer].gt)])
            fgrid = pomdp.belief[layer][column](gridx)
            fs = marginals(fgrid)
            marginal_stds = std.(fs)
            layer_col_unc += sum(marginal_stds)
        end
    end
    pomdp.map_uncertainty = layer_col_unc
    return layer_col_unc
end

function reward_information_gain(pomdp::CCSPOMDP)
    r::Float64 = pomdp.map_uncertainty
    if r < 0
        calculate_map_uncertainty(pomdp)
        return 1.0 # If its the first action, we don't know how much better we've done
        # Does this incentivize bad first actions?? Most likely yes, talk to Mansur
    end
    r -= calculate_map_uncertainty(pomdp)
    return r
end

function reward_suitability(pomdp::CCSPOMDP)
    total_grid_suitability = 0.
    for layer in 1:NUM_LAYERS
        gridx = pcu([pt.vertices[1] for pt in domain(pomdp.state.earth[layer].gt)])
        for pt in gridx
            # For each column gerate a distribution
            pt_score_samples = zeros(SUITABILITY_NSAMPLES)
            for column in pomdp.feature_names
                fgrid = pomdp.belief[layer][column]([pt])
                pt_score_samples .+= [score_component(column, rand(fgrid)[1]) for _ in 1:SUITABILITY_NSAMPLES]
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

function POMDPs.reward(pomdp::CCSPOMDP, state, action)
    action_cost = reward_action_cost(action)
    # println("action_cost $action_cost")
    
    information_gain = reward_information_gain(pomdp)
    # println("information_gain: $information_gain")
    
    suitability = reward_suitability(pomdp)
    # println("reward_suitability: $suitability")
    
    total_reward = action_cost + λ_1 * information_gain + λ_2 * suitability
    return total_reward
end


POMDPs.discount(pomdp::CCSPOMDP) = 0.95 


