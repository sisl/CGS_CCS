using CCSPOMDPs
using POMDPs
using POMDPModels
using POMDPSimulators
using POMCPOW
using Dates

TREE_QUERIES = 2
MAX_DEPTH = 2
MAX_STEPS = 3

pomdp = CCSPOMDPs.CCSPOMDP()
solver = POMCPOWSolver(tree_queries=TREE_QUERIES, max_depth=MAX_DEPTH)
planner = POMDPs.solve(solver, pomdp)

# Ask Mansur about max_steps
hr = HistoryRecorder(max_steps=MAX_STEPS, show_progress=true)
hist = simulate(hr, pomdp, planner)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename = "pomcpow_outputs/results_$timestamp.txt"

cumulative_discounted_reward = discounted_reward(hist)

open(filename, "w") do file
    for row in hist
        println(file, "a: $(row.a.id), r: $(row.r)")
    end
    
    println(file)
    println(file, "Cumulative Discounted Reward:")
    println(file, "    POMCPOW: $cumulative_discounted_reward")
end
