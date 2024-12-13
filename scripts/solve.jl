using CCSPOMDPs
using POMDPs
using POMDPModels
using POMDPSimulators
using POMCPOW
using Dates
using D3Trees
using JLD2

function run_solver(max_depth = 22, tree_queries=100, tree_in_info=true)
    println("Max Depth: $max_depth, Tree Queries: $tree_queries")
    pomdp = CCSPOMDPs.CCSPOMDP();
    solver = POMCPOWSolver(tree_queries=tree_queries,
                            max_depth=max_depth,
                            tree_in_info=tree_in_info,
                            # estimate_value=0.0,
                            enable_action_pw=false,
                            check_repeat_obs=false,);

    planner = POMDPs.solve(solver, pomdp);

    hr = HistoryRecorder();
    hist = simulate(hr, pomdp, planner)
    return hist;
end

function write_results(hist)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    dirname = "pomcpow_outputs/results_$timestamp"
    mkdir(dirname)
    filename = "$dirname/reward.txt"
    serialized_tree = "$dirname/tree.jld2"
    cumulative_discounted_reward = discounted_reward(hist)
    open(filename, "w") do file
        println(file, "Cumulative Discounted Reward: $cumulative_discounted_reward")
    end
    info = first(ainfo_hist(hist))
    tree = D3Tree(info[:tree])
    
    @save serialized_tree tree;
end

open("output.txt", "w") do io
    redirect_stdout(io) do
        @time hist = run_solver()
        write_results(hist);
    end
end