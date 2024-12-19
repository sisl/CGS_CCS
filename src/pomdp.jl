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
    function LayerFeatures(beliefs, layer::Int)
        layer_rtype = Base.rand(1:3)
        dfs::Vector{DataFrame} = []
        for feat in FEATURE_NAMES
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
mutable struct CCSState
    # Some measure of uncertainty over the whole map,
    # if < 0, uninitialized
    map_uncertainty::Float64 
    belief::Vector{Vector{Dict{Symbol, Any}}} # Every layer/feature combo has 3 GPs : for shale, siltstone, sandstone
    rocktype_belief::Vector{Distributions.Categorical{Float64, Vector{Float64}}}
    isterminated::Bool
    num_points::Int
    obs_pdf # Observation for that state and its pdf
end

struct ScaledEuclidean <: Distances.PreMetric
end

@inline function Distances._evaluate(::ScaledEuclidean,a::AbstractVector{T},b::AbstractVector{T}) where {T}
    ans = sqrt(sum(((a ./ SPACING) .- (b ./ SPACING)) .^ 2))
    return ans
end

@inline (dist::ScaledEuclidean)(a::AbstractArray,b::AbstractArray) = Distances._evaluate(dist,a,b)
@inline (dist::ScaledEuclidean)(a::Number, b::Number) = begin
    Euclidean()(a / SPACING, b / SPACING)
end

CCSAction = @NamedTuple{id::Symbol, geometry::Geometry}
struct CCSPOMDP <: POMDP{CCSState, @NamedTuple{id::Symbol, geometry::Geometry}, Any}
    initial_state::CCSState
    action_index::Dict
    earth::Vector{LayerFeatures}
    seismic_lines::Vector{Segment}
    well_logs::Vector{Point}

    initialize_earth(beliefs) = [LayerFeatures(beliefs, layer) for layer in 1:NUM_LAYERS]

    function initialize_belief()::Vector{Vector{Dict}}
        starter_beliefs::Vector{Vector{Dict}} = []
        for rocktype in 1:length(instances(RockType))
            rocktype_starter_beliefs::Vector{Dict} = []
            for layer in 1:NUM_LAYERS
                layer_beliefs = Dict()
                for column in FEATURE_NAMES
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
        lines = [Segment(Point(Base.rand(0.0:float(GRID_SIZE * SPACING)), 
                                Base.rand(0.0:float(GRID_SIZE * SPACING))), 
                         Point(Base.rand(0.0:float(GRID_SIZE * SPACING)),
                                Base.rand(0.0:float(GRID_SIZE * SPACING))))
                 for _ in 1:NUM_LINES]

        wells = [Point(Base.rand(0.0:float(GRID_SIZE * SPACING)), 
                        Base.rand(0.0:float(GRID_SIZE * SPACING))) 
                 for _ in 1:NUM_WELLS]
        
        gps = initialize_belief()
        earth = initialize_earth(gps)

        rtype_belief = [Distributions.Categorical(length(instances(RockType))) for _ in 1:NUM_LAYERS]
        
        return new(CCSState(-1.0, 
                            gps, 
                            rtype_belief, 
                            false, 
                            0, 
                            nothing), 
                    Dict(), # Will get filled the first time actionindex is called
                    earth,
                    lines,
                    wells) 
                    
    end

end
include("vis_utils.jl")

POMDPs.initialstate(pomdp::CCSPOMDP) = Deterministic(deepcopy(pomdp.initial_state))

function POMDPs.isterminal(pomdp::CCSPOMDP, state::CCSState)
    return state.isterminated
end

# TODO: Sanity check: level in tree to number of conditioning points
function POMDPs.gen(pomdp::CCSPOMDP, state, action, rng)
    sp = generate_nextstate(pomdp, state, action)
    o = update_sp_wobs(pomdp, state, action, sp)
    r = POMDPs.reward(pomdp, state, action, sp)
    return (sp=sp, o=o, r=r)
end

function generate_nextstate(pomdp::CCSPOMDP, state, action)
    # Construct next state based on action
    nextstate::CCSState = deepcopy(state) # Does this take too long?
    if action.id == :well_action
        nextstate.num_points += 1
    elseif action.id == :seismic_action
        nextstate.num_points += SEISMIC_N_POINTS
    elseif action.id == :terminate_action
        nextstate.isterminated = true
    end
    return nextstate
end

function update_sp_wobs(pomdp::CCSPOMDP, prevstate, action, state)
    components = []
    for layer in 1:NUM_LAYERS
        if action.id == :well_action # an observation with action_id == :well_action determines rock type
            rocktype = pomdp.earth[layer].layer_rocktype
            upd_prob = zeros(length(instances(RockType)))
            upd_prob[Int(rocktype)] = 1.0
            state.rocktype_belief[layer] = Distributions.Categorical(upd_prob)
        end
        for feature in FEATURE_NAMES
            x, y = get_x_y(pomdp, action.geometry, layer, feature)
            obs = observe(state, x, y, layer, feature, action.id)
            push!(components, obs)
        end
    end
    # println("Returning $components")
    state.obs_pdf = components
    return components
end

function ParticleFilters.obs_weight(pomdp::CCSPOMDP, state, action, state_prime, observation)
    cum_prob = 1.0
    for layer in 1:NUM_LAYERS
        for (findx, feature) in enumerate(FEATURE_NAMES)
            pdf_layer_feature = 0.0
            for rocktype in 1:length(instances(RockType))
                if state_prime.rocktype_belief[layer].p[rocktype] == 0.0
                    continue
                end
                f = state_prime.belief[rocktype][layer][feature]
                x, y = get_x_y(pomdp, action.geometry, layer, feature)
                log_prob = logpdf(f(x), observation[(layer - 1) * length(FEATURE_NAMES) + findx])
                pdf_layer_feature += state_prime.rocktype_belief[layer].p[rocktype] * exp(log_prob)
            end
            cum_prob *= pdf_layer_feature
        end
    end

    return cum_prob
end

function POMDPs.actions(pomdp::CCSPOMDP)::Vector{NamedTuple{(:id, :geometry), Tuple{Symbol, Geometry}}}
    well_actions = [(id=:well_action, geometry=x) for x in pomdp.well_logs]
    seismic_actions = [(id=:seismic_action, geometry=x) for x in pomdp.seismic_lines]
    # if you feel your solver is capable of handling many actions, but bad at recalling its own belief, you
    # can include observation actions by uncommenting the following line. This would add ~ 500 extra actions though,
    # so its generally not a good idea.
    observe_actions = [] # [(id=:observe_action, geometry=x.vertices[1]) for x in domain(pomdp.earth[1].gt)]
    terminate_action = [(id=:terminate_action, geometry=Point(0., 0.) )]
    return [well_actions; seismic_actions; observe_actions; terminate_action]
end

function POMDPs.actionindex(pomdp::CCSPOMDP, action)
    if action in keys(pomdp.action_index)
        return pomdp.action_index[action]
    end
    all_actions = POMDPs.actions(pomdp)
    for (i, action) in enumerate(all_actions)
        pomdp.action_index[action] = i
    end
    return pomdp.action_index[action]
end
function get_x_y(pomdp::CCSPOMDP, point::Point, layer::Int, column::Symbol)
    x = pcu(point)
    y = pomdp.earth[layer].gt[point, column]
    return x, y
end 
function get_x_y(pomdp::CCSPOMDP, geom::Segment, layer::Int, column::Symbol)
    p1 = geom.vertices[1]
    p2 = geom.vertices[2]
    points = [p1 * (1 - t) + p2 * t for t in range(0, stop=1, length=SEISMIC_N_POINTS)]
    x = pcu(points)
    y = [pomdp.earth[layer].gt[pt, column][1] for pt in points]
    return x, y
end

function observe(state::CCSState, x::Vector{Vector{Float64}}, y::Vector{Float64}, layer::Int, column::Symbol, action_id::Symbol)
    # We return a mixture model of the GP conditioned on the rocktype
    obs_conditioned_on_rocktype::Vector{Vector{Float64}} = Vector(undef, length(instances(RockType)))
    dummy_distr = MvNormal(1, 1.0)
    unc = ACTION_UNCERTAINTY[(action_id, column)]
    
    for rocktype in 1:length(instances(RockType))
        if state.rocktype_belief[layer].p[rocktype] == 0.0
            obs_conditioned_on_rocktype[rocktype] = rand(dummy_distr) # a short circuit when rocktype has probability 0
        end
        if unc >= 0.0
            state.belief[rocktype][layer][column] = posterior(state.belief[rocktype][layer][column](x, unc), y)
        end

        fx = state.belief[rocktype][layer][column](x)
        obs_conditioned_on_rocktype[rocktype] = rand(fx)
    end

    rand_ind = rand(state.rocktype_belief[layer])
    
    return obs_conditioned_on_rocktype[rand_ind]
end

function reward_action_cost(action::NamedTuple{(:id, :geometry), Tuple{Symbol, Geometry}})
    if action.id == :well_action
        return -WELL_COST
    elseif action.id == :seismic_action
        return -SEISMIC_LINE_COST
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
        if (1 / 1 + exp(-value)) < 0.55
            return 1
        else
            return 5
        end
    end
end

"""
Computes the information gain and suitability reward components
While this would be more readable as two separate functions, the pattern of indexing
(layer, column, rocktype) is the same for both, so for performance reasons, we combine them.

Information gain is measured as the mean of scaled marginal standard deviations of the GP
# Eqn for variance of mixture distribution: https://en.wikipedia.org/wiki/Mixture_distribution

Suitability is measured as the sum of the number of points that we are confident are suitable or unsuitable.
"""
function reward_information_suitability(state::CCSState) 
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
    scaling_factor = Dict(feat => mean([PRIOR_BELIEF[(feat, rtype)][2] for rtype in instances(RockType)]) for feat in FEATURE_NAMES)

    for layer in 1:NUM_LAYERS
        # suitability
        fill!(sample_values, 0.0)
        fill!(prob_mask, 0.0)

        for column in FEATURE_NAMES
            # information_gain
            fill!(scaled_var_mtx, 0.0)
            fill!(all_rock_mean, 0.0)

            for rocktype in 1:length(instances(RockType))
                belief_prob = state.rocktype_belief[layer].p[rocktype]
                if belief_prob == 0.0
                    continue
                end
                all_rock_mean .+= mean(state.belief[rocktype][layer][column], GRIDX) .* belief_prob
            end
            
            all_rock_mean = all_rock_mean .^ 2 
            # suitability
            prev_end = 0
            for rocktype in 1:length(instances(RockType))
                # both
                belief_prob = state.rocktype_belief[layer].p[rocktype]
                if belief_prob == 0.0
                    continue
                end
                # println("Marginals computation")
                # @time 
                ms = marginals(state.belief[rocktype][layer][column](GRIDX))
                mg_stds = std.(ms)
                mg_means = mean.(ms)

                # information gain
                var_compontent = ((mg_stds .^ 2) .+ (mg_means) .^ 2 .- all_rock_mean) .* belief_prob
                scaled_var_mtx .+= reshape(var_compontent, GRID_SIZE, GRID_SIZE)' 

                # suitability
                rocktype_nsamples = Int(floor(belief_prob * SUITABILITY_NSAMPLES))
                norms = MvNormal(mg_means, diagm(mg_stds))
                incr = [score_component(column, n) for n in rand(norms) for _ in 1:rocktype_nsamples]
                sample_values[npts * prev_end + 1: npts * (prev_end + rocktype_nsamples)] .+= incr
                prob_mask[prev_end + 1:prev_end + rocktype_nsamples] .= rocktype
                prev_end += rocktype_nsamples
            end
            # println("w/o scaling, For column $column, with scaling factor: $(sqrt(scaling_factor)), mean uncertainty after scaling: $(mean(sqrt.(scaled_var_mtx ./ scaling_factor)))")
            layer_col_unc += mean(sqrt.(scaled_var_mtx ./ scaling_factor[column]))
        end
        # suitability
        sample_values ./= length(FEATURE_NAMES)
        sample_matr = reshape(sample_values, length(GRIDX), SUITABILITY_NSAMPLES)
        bits_matr = sample_matr .> SUITABILITY_THRESHOLD
        suitable_pts = (Statistics.mean(bits_matr .* prob_mask', dims=2) .>= SUITABILITY_CONF_THRESHOLD)
        total_grid_suitability += sum(suitable_pts)
        bits_matr = .!bits_matr
        unsuitable_pts = (Statistics.mean(bits_matr .* prob_mask', dims=2) .>= SUITABILITY_CONF_THRESHOLD)
        total_grid_suitability += SUITABILITY_BIAS * sum(unsuitable_pts)
    end
    # println("Total Layer Col uncertainty: $layer_col_unc")
    state.map_uncertainty = layer_col_unc

    return layer_col_unc, total_grid_suitability
end

function POMDPs.reward(pomdp::CCSPOMDP, state, action, state_prime)
    if action.id == :terminate_action
        return 0.0
    end
    action_cost = reward_action_cost(action)
    
    if state.map_uncertainty < 0.0
        orig_uncertainty, _ = reward_information_suitability(state)
    else
        orig_uncertainty = state.map_uncertainty
    end
    new_uncertainty, suitability = reward_information_suitability(state_prime)

    # println("Action: $(action.id), \nzzzAction Cost: $action_cost, \nOrig_uncertainty: $(λ_1*(orig_uncertainty))\nNew_uncertainty: $(λ_1*(new_uncertainty))\nzzzUncertainty Change: $(λ_1*(orig_uncertainty - new_uncertainty)), \nzzzSuitability: $(λ_2*suitability)")
    return action_cost + λ_1 * (orig_uncertainty - new_uncertainty) + λ_2 * suitability
end


POMDPs.discount(pomdp::CCSPOMDP) = 0.99