using JLD2
using D3Trees

treename = "pomcpow_outputs/results_2024-12-15_16-31-10/tree.jld2"

@load treename tree

inchrome(tree)