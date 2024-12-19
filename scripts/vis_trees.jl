using JLD2
using D3Trees

treename = "pomcpow_outputs/results_2024-12-19_13-34-11/tree.jld2"

@load treename tree

inchrome(tree)