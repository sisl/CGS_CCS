using JLD2
using D3Trees

treename = "pomcpow_outputs/results_2024-12-19_14-14-22/tree.jld2"

@load treename tree

inchrome(tree)