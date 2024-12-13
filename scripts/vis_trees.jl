using JLD2
using D3Trees

treename = "pomcpow_outputs/results_2024-12-10_13-15-59/tree.jld2"

@load treename tree

inchrome(tree)