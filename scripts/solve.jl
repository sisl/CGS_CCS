using CCSPOMDPs
using POMDPModels
using POMDPSimulators
using POMCPOW

# include("CGS_POMDP.jl")

pomdp = CCSPOMDP()
solver = POMCPOWSolver(criterion=MaxUCB(20.0))
planner = solve(solver, pomdp)

hr = HistoryRecorder(max_steps=100)
hist = simulate(hr, pomdp, planner)

for (s, b, a, r, sp, o) in hist
    @show s, a, r, sp
end