using POMDPs
using POMDPModels
using POMDPSimulators
using POMCPOW

include("CGS_POMDP.jl")

pomdp = CGSPOMDP()

# Select a solver (e.g., a value iteration solver for discrete models)
solver = POMCPOWSolver(criterion=MaxUCB(20.0))

# Solve the POMDP to obtain a policy
planner = solve(solver, pomdp)

hr = HistoryRecorder(max_steps=100)
hist = simulate(hr, pomdp, planner)

# Inspect the simulation history
for (s, b, a, r, sp, o) in hist
    @show s, a, r, sp
end