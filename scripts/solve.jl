using CCSPOMDPs
using POMDPs
using POMDPModels
using POMDPSimulators
using POMCPOW
using Dates

pomdp = CCSPOMDPs.CCSPOMDP()
solver = POMCPOWSolver(tree_queries=4, max_depth=2)
planner = POMDPs.solve(solver, pomdp)

hr = HistoryRecorder(max_steps=4, show_progress=true)
hist = simulate(hr, pomdp, planner)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename = "pomcpow_outputs/results_$timestamp.txt"

cumulative_discounted_reward = discounted_reward(hist)

open(filename, "w") do file
    for (s, a, sp, r, info) in hist
        println(file, "a: $(a.id), r: $r")
    end
    
    println(file)
    println(file, "Cumulative Discounted Reward:")
    println(file, "    POMCPOW: $cumulative_discounted_reward")
end
