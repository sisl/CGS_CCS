include("mesh_utils.jl")

GRID = CartesianGrid((GRID_SIZE, GRID_SIZE), (0., 0.), (SPACING, SPACING))
GRIDX = pcu([pt.vertices[1] for pt in GRID])

"""
LayerFeatures struct

Holds a GeoTable for each Layer, with measurements for each feature
use viewer(gt) to visualize a Layer
"""
struct LayerFeatures
    gt::GeoTable
    df::DataFrame
    layer_rocktype::RockType
    function LayerFeatures(feats::Vector{Symbol}, beliefs, layer::Int)
        layer_rtype = Base.rand(1:3)
        dfs::Vector{DataFrame} = []
        for feat in feats
            f = beliefs[layer_rtype][layer][feat]
            simul = rand(f(GRIDX))
            df = DataFrame(feat => simul)
            push!(dfs, df)
        end
        all_feature_df = hcat(dfs...)
        all_feature_gt = georef(all_feature_df, GRID)
        
        return new(all_feature_gt, all_feature_df, RockType(layer_rtype))
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

struct ScaledEuclidean <: Distances.PreMetric
end

@inline function Distances._evaluate(::ScaledEuclidean,a::AbstractVector{T},b::AbstractVector{T}) where {T}
    # @boundscheck if length(a) != length(b)
    #     throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    # end
    ans = sqrt(sum(((a ./ SPACING) .- (b ./ SPACING)) .^ 2))
    # if Base.rand(1:2500000) == 42
    #     println("Dist betweeen $a and $b is $ans")
    # end
    return ans
end

@inline (dist::ScaledEuclidean)(a::AbstractArray,b::AbstractArray) = Distances._evaluate(dist,a,b)
@inline (dist::ScaledEuclidean)(a::Number, b::Number) = begin
    # println("Dist betweeen $a and $b is $(Euclidean()(a, b))") # Always checks between 1.0 and 1.0 for some reason
    Euclidean()(a / SPACING, b / SPACING)
end

mutable struct CCSPOMDP <: POMDP{CCS_State, @NamedTuple{id::Symbol, geometry::Geometry}, Any}
    state::CCS_State
    feature_names::Vector{Symbol}
    map_uncertainty::Float64 # Some measure of uncertainty over the whole map
    belief::Vector{Vector{Dict{Symbol, Any}}} # Every layer/feature combo has 3 GPs : for shale, siltstone, sandstone
    action_index::Dict
    rocktype_belief::Vector{Distributions.Categorical{Float64, Vector{Float64}}}
    terminal_state::CCS_State

    initialize_earth(beliefs, feat_names) = [LayerFeatures(feat_names, beliefs, layer) for layer in 1:NUM_LAYERS]

    function initialize_belief(feature_names::Vector{Symbol})::Vector{Vector{Dict}}
        starter_beliefs::Vector{Vector{Dict}} = []
        for rocktype in 1:length(instances(RockType))
            rocktype_starter_beliefs::Vector{Dict} = []
            for layer in 1:NUM_LAYERS
                layer_beliefs = Dict()
                for column in feature_names
                    if column == :z
                        # Simplifying assumption: z is a linear function of layer with noise
                        shift, scale = (500 * layer, 200 * 200)
                    else
                        shift, scale = PRIOR_BELIEF[(column, RockType(rocktype))]
                    end
                    layer_beliefs[column] = GP(shift, ScaledKernel(MaternKernel(ν=2.5, metric=ScaledEuclidean()), scale))
                end
                push!(rocktype_starter_beliefs, layer_beliefs)
            end
            push!(starter_beliefs, rocktype_starter_beliefs)
        end
        return starter_beliefs
    end

    function CCSPOMDP()
        feature_names = FEATURE_NAMES
        lines = [Segment(Point(Base.rand(0.0:float(GRID_SIZE * SPACING)), 
        Base.rand(0.0:float(GRID_SIZE * SPACING))), 
        Point(Base.rand(0.0:float(GRID_SIZE * SPACING)),
        Base.rand(0.0:float(GRID_SIZE * SPACING))))
        for _ in 1:NUM_LINES]
        wells = [Point(Base.rand(0.0:float(GRID_SIZE * SPACING)), 
        Base.rand(0.0:float(GRID_SIZE * SPACING))) for _ in 1:NUM_WELLS]
        
        gps = initialize_belief(feature_names)
        earth = initialize_earth(gps, feature_names)

        rtype_belief = [Distributions.Categorical(length(instances(RockType))) for _ in 1:NUM_LAYERS]
        
        pomdp = new(CCS_State(earth, lines, wells), 
                        feature_names, 
                        -1.0, # signals unknown map uncertainty
                        gps,
                        Dict(), # TODO: Initialize action and actionindex here!
                        rtype_belief,
                        CCS_State(Vector{LayerFeatures}(), Vector{Segment}(), Vector{Point}())
                        )

        return pomdp
    end

end
include("vis_utils.jl")

POMDPs.states(pomdp::CCSPOMDP) = [pomdp.state, pomdp.terminal_state]

POMDPs.initialstate(pomdp::CCSPOMDP) = SparseCat([pomdp.state, pomdp.terminal_state], [1.0, 0.0])

POMDPs.stateindex(pomdp::CCSPOMDP, state::CCS_State) = length(pomdp.state.earth) > 0 ? 1 : 2

function POMDPs.transition(pomdp::CCSPOMDP, state, action)
    if action.id == :terminate_action
        return SparseCat([pomdp.state, pomdp.terminal_state], [0.0, 1.0])
    else
        return SparseCat([pomdp.state, pomdp.terminal_state], [1.0, 0.0])
    end
end

function POMDPs.actions(pomdp::CCSPOMDP)::Vector{NamedTuple{(:id, :geometry), Tuple{Symbol, Geometry}}}
    well_actions = [(id=:well_action, geometry=x) for x in pomdp.state.well_logs]
    seismic_actions = [(id=:seismic_action, geometry=x) for x in pomdp.state.seismic_lines]
    # Removing observe actions for the time being.
    observe_actions = [] # [(id=:observe_action, geometry=x.vertices[1]) for x in domain(pomdp.state.earth[1].gt)]
    terminate_action = [(id=:terminate_action, geometry=Segment(Point(0., 0.), Point(0., 0.)))]
    return [well_actions; seismic_actions; observe_actions; terminate_action]
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
    # We return a mixture model of the GP conditioned on the rocktype
    obs_conditioned_on_rocktype = Vector{MvNormal}(undef, length(instances(RockType)))
    x = pcu(point)
    y = pomdp.state.earth[layer].gt[point, column]
    unc = ACTION_UNCERTAINTY[(action_id, column)]
    
    for rocktype in 1:length(instances(RockType))
        if pomdp.rocktype_belief[layer].p[rocktype] == 0.0
            MvNormal(1, 1.0) # a short circuit when rocktype has probability 0
        end
        f = pomdp.belief[rocktype][layer][column]
        if unc < 0 # Feature belief not changed by action
            mean_cond = mean(f(x))
            cov_cond = cov(f(x))
        else
            p_fx = posterior(f(x, unc), y)
            pomdp.belief[rocktype][layer][column] = p_fx
            
            mean_cond = mean(p_fx(x))
            cov_cond = cov(p_fx(x))
        end
        obs_conditioned_on_rocktype[rocktype] = MvNormal(mean_cond, cov_cond)
    end
    return Distributions.MixtureModel(obs_conditioned_on_rocktype, pomdp.rocktype_belief[layer].p)
end

function observe(pomdp::CCSPOMDP, geom::Segment, layer::Int, column::Symbol, action_id::Symbol)
    p1 = geom.vertices[1]
    p2 = geom.vertices[2]
    points = [p1 * (1 - t) + p2 * t for t in range(0, stop=1, length=SEISMIC_N_POINTS)]
    obs_conditioned_on_rocktype = Vector{MvNormal}(undef, length(instances(RockType)))
    x = pcu(points)
    y = [pomdp.state.earth[layer].gt[pt, column][1] for pt in points]
    unc = ACTION_UNCERTAINTY[(action_id, column)]

    for rocktype in 1:length(instances(RockType))
        if pomdp.rocktype_belief[layer].p[rocktype] == 0.0 || action_id == :terminate_action
            MvNormal(1, 1.0) # a short circuit when rocktype has probability 0
        end
        f = pomdp.belief[rocktype][layer][column]
        if unc < 0 # Feature belief not changed by action
            mean_cond = mean(f(x))
            cov_cond = cov(f(x))
        else
            p_fx = posterior(f(x, unc), y)
            pomdp.belief[rocktype][layer][column] = p_fx
            
            mean_cond = mean(p_fx(x))
            cov_cond = cov(p_fx(x))
        end
        obs_conditioned_on_rocktype[rocktype] = MvNormal(mean_cond, cov_cond)
    end
    return Distributions.MixtureModel(obs_conditioned_on_rocktype, pomdp.rocktype_belief[layer].p)
end

function POMDPs.observation(pomdp::CCSPOMDP, action, state)
    component_distributions::Vector{MixtureModel{Multivariate, Distributions.Continuous, MvNormal, Distributions.Categorical{Float64, Vector{Float64}}}} = []
    for layer in 1:NUM_LAYERS
        if action.id == :well_action # an observation with action_id == :well_action determines rock type
            rocktype = pomdp.state.earth[layer].layer_rocktype
            upd_prob = zeros(length(instances(RockType)))
            upd_prob[Int(rocktype)] = 1.0
            pomdp.rocktype_belief[layer] = Distributions.Categorical(upd_prob)
        end
        for feature in pomdp.feature_names
            push!(component_distributions, observe(pomdp, action.geometry, layer, feature, action.id))
        end
    end

    return product_distribution(component_distributions)
end

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
    elseif feature == :injectivity
        if value < 0.25
            return 0
        elseif value < 0.5
            return 1
        elseif value < 1
            return 2
        elseif value < 1.5
            return 3
        elseif value < 2
            return 4
        else
            return 5
        end
    elseif feature == :salinity # recall ppm * 1000
        if value < 10
            return 0
        elseif value < 30
            return 2
        elseif value < 50
            return 5
        elseif value < 100
            return 4
        elseif value < 200
            return 3
        else
            return 1
        end
    elseif feature == :bottomSeal
        if value < 0.1
            return 1
        else
            return 5
        end
    end
end

function mean_and_var(f, x)
    C_xcond_x = cov(f.prior, f.data.x, x)
    m_post = mean(f.prior, x) + C_xcond_x' * f.data.α
    C_post_diag = var(f.prior, x) - vec(sum(abs2, f.data.C.U' \ C_xcond_x; dims=1))
    return (m_post, C_post_diag)
end

function my_marginals(f, x)
    m, c = mean_and_var(f, x)
    return Normal.(m, sqrt.(c))
end

"""
Computes the information gain and suitability reward components
While this would be more readable as two separate functions, the pattern of indexing
(layer, column, rocktype) is the same for both, so for performance reasons, we combine them.

Information gain is measured as the mean of scaled marginal standard deviations of the GP
# Eqn for variance of mixture distribution: https://en.wikipedia.org/wiki/Mixture_distribution

Suitability is measured as the sum of the number of points that we are confident are suitable or unsuitable.
"""
function reward_information_gain_suitability(pomdp::CCSPOMDP)
    original_uncertainty::Float64 = pomdp.map_uncertainty
    
    # Both info and suitability
    npts = length(GRIDX)

    # information_gain
    layer_col_unc = 0.0
    scaled_var_mtx = zeros(GRID_SIZE, GRID_SIZE)
    all_rock_mean = zeros(npts)
    
    # suitability
    total_grid_suitability = 0. # This method stratifies suitability sampling on rock type.
    sample_values = zeros(npts * SUITABILITY_NSAMPLES)
    prob_mask = zeros(SUITABILITY_NSAMPLES)

    for layer in 1:NUM_LAYERS
        # suitability
        fill!(sample_values, 0.0)
        fill!(prob_mask, 0.0)

        for column in pomdp.feature_names
            # information_gain
            fill!(scaled_var_mtx, 0.0)
            fill!(all_rock_mean, 0.0)

            for rocktype in 1:length(instances(RockType))
                belief_prob = pomdp.rocktype_belief[layer].p[rocktype]
                if belief_prob == 0.0
                    continue
                end
                all_rock_mean .+= mean(pomdp.belief[rocktype][layer][column], GRIDX) .* belief_prob
            end
            
            all_rock_mean ./= 2.
            # suitability
            prev_end = 0
            for rocktype in 1:length(instances(RockType))
                # both
                belief_prob = pomdp.rocktype_belief[layer].p[rocktype]
                if belief_prob == 0.0
                    continue
                end
                ms = marginals(pomdp.belief[rocktype][layer][column](GRIDX))
                mg_stds = std.(ms)
                mg_means = mean.(ms)

                # information gain
                var_compontent = ((mg_stds .^ 2) .+ (mg_means .- all_rock_mean) .^ 2) .* belief_prob
                scaled_var_mtx .+= reshape(var_compontent, GRID_SIZE, GRID_SIZE)' ./ PRIOR_BELIEF[(column, RockType(rocktype))][2]

                # suitability
                rocktype_nsamples = Int(floor(belief_prob * SUITABILITY_NSAMPLES))
                norms = [Normal(μ, σ) for (μ, σ) in zip(mg_means, mg_stds)]
                incr = [score_component(column, rand(N)) for N in norms for _ in 1:rocktype_nsamples]
                sample_values[npts * prev_end + 1: npts * (prev_end + rocktype_nsamples)] .+= incr
                prob_mask[prev_end + 1:prev_end + rocktype_nsamples] .= rocktype
                prev_end += rocktype_nsamples
            end
            layer_col_unc += mean(sqrt.(scaled_var_mtx))
        end
        # suitability
        sample_values ./= length(pomdp.feature_names)
        sample_matr = reshape(sample_values, length(GRIDX), SUITABILITY_NSAMPLES)
        bits_matr = sample_matr .> SUITABILITY_THRESHOLD
        suitable_pts = (Statistics.mean(bits_matr .* prob_mask', dims=2) .>= SUITABILITY_CONF_THRESHOLD)
        total_grid_suitability += sum(suitable_pts)
        bits_matr = .!bits_matr
        unsuitable_pts = (Statistics.mean(bits_matr .* prob_mask', dims=2) .>= SUITABILITY_CONF_THRESHOLD)
        total_grid_suitability += SUITABILITY_BIAS * sum(unsuitable_pts)
    end

    pomdp.map_uncertainty = layer_col_unc

    return original_uncertainty - layer_col_unc, total_grid_suitability
end

function POMDPs.reward(pomdp::CCSPOMDP, state, action)
    # println("reward $(action.id)")
    action_cost = reward_action_cost(action)
    
    information_gain, suitability = reward_information_gain_suitability(pomdp)

    total_reward = action_cost + λ_1 * information_gain + λ_2 * suitability
    return total_reward
end

# function POMDPs.gen(pomdp::CCSPOMDP, state, action, rng)
#     println("Observation: ")
#     @time oc = rand(POMDPs.observation(pomdp, action, state))
#     println("Reward: ")
#     @time rew = POMDPs.reward(pomdp, state, action)
#     return (sp = pomdp.state,
#             o = oc,
#             r = rew)
# end

POMDPs.discount(pomdp::CCSPOMDP) = 0.95 