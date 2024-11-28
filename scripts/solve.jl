import CCSPOMDPs
using POMDPs
using POMDPModels
using POMDPSimulators
using POMCPOW

pomdp = CCSPOMDPs.CCSPOMDP()
solver = POMCPOWSolver(tree_queries=4, max_depth=2) # Note time constraint makes this a bad solver
planner = POMDPs.solve(solver, pomdp)

hr = HistoryRecorder(max_steps=4, show_progress=true)
hist = simulate(hr, pomdp, planner)

for (s, b, a, r, sp, o) in hist
    @show s, a, r, sp
end