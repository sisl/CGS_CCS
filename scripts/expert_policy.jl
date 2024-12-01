import CCSPOMDPs
using POMDPs
using POMDPModels
using POMDPPolicies
using POMDPModelTools
using POMDPSimulators
using Meshes


struct GridCell
    region::Box
    probability::Float64
end

println("$(CCSPOMDPs.SPACING)")

struct CGSExpertPolicy{P<:POMDP} <: Policy
    pomdp::P
    grid::Vector{GridCell}
    all_actions::Set
    budget::Float64
    grid_size::Int
    function CGSExpertPolicy(pomdp::P) where {P<:POMDP}
        grid = Vector{GridCell}()
        all_actions_ = Set([action for action in actions(pomdp) if action.id != :terminate_action])
        budget = 25.0
        grid_size = 10
        for i in 0:grid_size - 1
            for j in 0:grid_size - 1
                region = Box((i * SPACING, j * SPACING), ((i + 1) * CCSPOMDPs.SPACING, (j + 1) * CCSPOMDPs.SPACING))
                probability = 1.0
                push!(grid, GridCell(region, probability))
            end
        end
        return new{P}(pomdp, grid, all_actions_, budget, grid_size)
    end 
end

function POMDPs.action(p::CGSExpertPolicy, b)
    chosen_action = rand(p.all_actions)
    p.budget -= CCSPOMDPs.reward_action_cost(chosen_action)

    if p.budget <= 0
        return actions(p.pomdp)[end] # terminate action
    end

    return chosen_action
end


pomdp = CCSPOMDPs.CCSPOMDP()

expert_pol = CGSExpertPolicy(pomdp);

rollout_sim = RolloutSimulator(max_steps=1);
expert_reward = simulate(rollout_sim, pomdp, expert_pol, DiscreteUpdater(pomdp));

@show expert_reward;