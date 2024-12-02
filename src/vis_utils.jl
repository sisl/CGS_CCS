# Visualization functions.
"""
Visualize the uncertainty in the belief of a layer and feature.
"""
function visualize_uncertainty(pomdp::CCSPOMDP, layer::Int, column::Symbol)
    var_mtx = zeros(GRID_SIZE, GRID_SIZE)
    all_rock_mean = zeros(length(GRIDX))
    for rocktype in 1:length(instances(RockType))
        if pomdp.rocktype_belief[layer].p[rocktype] == 0.0
            continue
        end
        ms = marginals(pomdp.belief[rocktype][layer][column](GRIDX))
        all_rock_mean += mean.(ms) * pomdp.rocktype_belief[layer].p[rocktype]
    end

    # Eqn for mixture distribution: https://en.wikipedia.org/wiki/Mixture_distribution
    for rocktype in 1:length(instances(RockType))
        if pomdp.rocktype_belief[layer].p[rocktype] == 0.0
            continue
        end
        ms = marginals(pomdp.belief[rocktype][layer][column](GRIDX))
        mg_stds = std.(ms)
        mg_means = mean.(ms)
        var_compontent = ((mg_stds .^ 2) + (mg_means - all_rock_mean) .^ 2) * pomdp.rocktype_belief[layer].p[rocktype]
        var_mtx .+= reshape(var_compontent, GRID_SIZE, GRID_SIZE)'
    end

    xaxisvalues = yaxisvalues = (1:GRID_SIZE) .* SPACING
    heatmap(xaxisvalues, 
            yaxisvalues,
            sqrt.(var_mtx),
            color=:viridis,
            xlabel="Grid X (m)",
            ylabel="Grid Y (m)",
            title="Uncertainty Layer $layer Feature $column",
            # clims=(0, maximum(mg_stds))
            )
end

function visualize_uncertainty(pomdp::CCSPOMDP, layer::Int, column::Symbol, supplementary_points::Vector)
    var_mtx = zeros(GRID_SIZE, GRID_SIZE)
    all_rock_mean = zeros(length(GRIDX))
    for rocktype in 1:length(instances(RockType))
        if pomdp.rocktype_belief[layer].p[rocktype] == 0.0
            continue
        end
        ms = marginals(pomdp.belief[rocktype][layer][column](GRIDX))
        all_rock_mean += mean.(ms) * pomdp.rocktype_belief[layer].p[rocktype]
    end

    # Eqn for mixture distribution: https://en.wikipedia.org/wiki/Mixture_distribution
    for rocktype in 1:length(instances(RockType))
        if pomdp.rocktype_belief[layer].p[rocktype] == 0.0
            continue
        end
        ms = marginals(pomdp.belief[rocktype][layer][column](GRIDX))
        mg_stds = std.(ms)
        println("Standard deviations of supplementary_points and 3 regular points: ", mg_stds[end - length(supplementary_points) - 3:end])
        mg_means = mean.(ms)
        var_compontent = ((mg_stds .^ 2) + (mg_means - all_rock_mean) .^ 2) * pomdp.rocktype_belief[layer].p[rocktype]
        var_mtx .+= reshape(var_compontent[1:end - length(supplementary_points)], GRID_SIZE, GRID_SIZE)'
    end

    xaxisvalues = yaxisvalues = (1:GRID_SIZE) .* SPACING
    heatmap(xaxisvalues, 
            yaxisvalues,
            sqrt.(var_mtx),
            color=:viridis,
            xlabel="Grid X (m)",
            ylabel="Grid Y (m)",
            title="Uncertainty Layer $layer Feature $column",
            # clims=(0, maximum(mg_stds))
            )
end

visualize_gt(pomdp::CCSPOMDP, layer::Int) = viewer(pomdp.state.earth[layer].gt)