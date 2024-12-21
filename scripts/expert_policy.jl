import CCSPOMDPs
using POMDPs
using POMDPModels
using POMDPPolicies
using POMDPModelTools
using POMDPSimulators
using Meshes
using BeliefUpdaters
using Statistics
using ProgressMeter

START_BUDGET = 150.0 * 6 # Basically whatever the POMDP spends

mutable struct GridCell
    region::Box
    probability::Float64
end

mutable struct CGSExpertPolicy{P<:POMDP} <: Policy
    pomdp::P
    grid::Vector{GridCell}
    budget::Float64
    grid_size::Int
    function CGSExpertPolicy(pomdp::P) where {P<:POMDP}
        grid = Vector{GridCell}()
        budget = START_BUDGET
        grid_size = 10
        scale_factor = (CCSPOMDPs.GRID_SIZE / grid_size) * CCSPOMDPs.SPACING
        for i in 0:grid_size - 1
            for j in 0:grid_size - 1
                region = Box((i * scale_factor, j * scale_factor), ((i + 1) * scale_factor, (j + 1) * scale_factor))
                probability = 1.0
                push!(grid, GridCell(region, probability))
            end
        end
        return new{P}(pomdp, grid, budget, grid_size)
    end 
end

function POMDPs.action(p::CGSExpertPolicy, b)
    all_actions = actions(p.pomdp)
    if rand(1:2) == 1
        action_choice = :seismic_action
    else
        action_choice = :well_action
    end
    probs = [0. for _ in all_actions]
    for (i, action) in enumerate(all_actions)
        for cell in p.grid
            if intersects(cell.region, action.geometry)
                probs[i] += cell.probability
            end
        end
        if action.id != action_choice
            probs[i] = 0
        end
    end
    chosen_action = rand(SparseCat(all_actions, probs))
    for cell in p.grid
        if intersects(cell.region, chosen_action.geometry)
            cell.probability *= 0.9
        end
    end
    p.budget += CCSPOMDPs.reward_action_cost(chosen_action)

    if p.budget <= 0
        return actions(p.pomdp)[end] # terminate action
    end
    println("Chosen Action: $chosen_action")
    return chosen_action
end


function runsims(nsims::Int)
    rewards = [0.0 for _ in 1:nsims]

    @showprogress for i in 1:nsims
        pomdp = CCSPOMDPs.CCSPOMDP();

        expert_pol = CGSExpertPolicy(pomdp);

        rollout_sim = RolloutSimulator(max_steps=20);
        expert_reward = simulate(rollout_sim, pomdp, expert_pol, NothingUpdater())

        rewards[i] = expert_reward
    end

    println("Expert Policy Average Reward: $(mean(rewards)), Std Dev: $(std(rewards))")
end

nsims = 1
runsims(nsims)