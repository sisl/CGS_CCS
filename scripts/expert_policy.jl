import CCSPOMDPs
using POMDPs
using POMDPModels
using POMDPPolicies
using POMDPSimulators
# using BeliefUpdaters


# struct CGSExpertPolicy{P<:POMDP} <: Policy
#     pomdp::P
# end

# function POMDPs.action(p::GreedyPolicy, b)
#     max_value = -Inf
#     as = actions(p.pomdp)
#     best_a = first(as)
#     for a in as
#         action_val = 0.0
#         for (state, bel) in weighted_iterator(b)
#             action_val += bel*reward(p.pomdp, state, a)
#         end
        
#         if action_val > max_value
#             best_a = a
#             max_value = action_val
#         end
#     end
    
#     return best_a
# end


pomdp = CCSPOMDPs.CCSPOMDP()

rand_policy = RandomPolicy(pomdp);
# expert_pol = CGSExpertPolicy(pomdp);

rollout_sim = RolloutSimulator(max_steps=1);
# expert_reward = simulate(rollout_sim, pomdp, expert_pol, DiscreteUpdater(pomdp));
rand_reward = simulate(rollout_sim, pomdp, rand_policy);

# @show expert_reward;
@show rand_reward;