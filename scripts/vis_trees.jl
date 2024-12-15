using JLD2
using D3Trees

treename = "pomcpow_outputs/results_2024-12-13_11-20-15/tree.jld2"

@load treename tree

inchrome(tree)