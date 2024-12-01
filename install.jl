using Pkg

dependencies = ["AbstractGPs",
                "AbstractGPsMakie",
                "CairoMakie",
                "DataFrames",
                "DataStructures",
                "Distributions",
                "GLMakie",
                "GeoTables",
                "Meshes",
                "POMDPModelTools",
                "Infiltrator",
                "LinearAlgebra",
                "POMCPOW",
                "POMDPModels",
                "POMDPPolicies",
                "POMDPSimulators",
                "POMDPs",
                "ParticleFilters",
                "Plots",
                "Random",
                "Reexport",
                "Statistics",
                "Unitful"]

Pkg.add(dependencies)