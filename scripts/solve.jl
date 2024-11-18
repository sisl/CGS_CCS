import CCSPOMDPs
using POMDPs
using POMDPModels
using POMDPSimulators
using POMCPOW

pomdp = CCSPOMDPs.CCSPOMDP()
solver = POMCPOWSolver(max_time=10, criterion=MaxUCB(20.0)) # Note time constraint makes this a bad solver
planner = POMDPs.solve(solver, pomdp)

hr = HistoryRecorder(max_steps=5)
hist = simulate(hr, pomdp, planner)

for (s, b, a, r, sp, o) in hist
    @show s, a, r, sp
end