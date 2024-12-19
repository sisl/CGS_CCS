using JLD2
using D3Trees

treename = "pomcpow_outputs/results_2024-12-17_14-02-20/tree.jld2"

@load treename tree

inchrome(tree)