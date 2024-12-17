using CCSPOMDPs
using POMDPs
using POMDPModels
using POMDPSimulators
using POMCPOW
using Dates
using D3Trees
using JLD2

function get_date_dirname()
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    dirname = "pomcpow_outputs/results_$timestamp"
end
function run_solver(max_depth = 22, tree_queries=1000, tree_in_info=true)
    println("Max Depth: $max_depth, Tree Queries: $tree_queries")
    pomdp = CCSPOMDPs.CCSPOMDP();
    solver = POMCPOWSolver(tree_queries=tree_queries,
                            max_depth=max_depth,
                            tree_in_info=tree_in_info,
                            estimate_value=0.0,
                            enable_action_pw=false,
                            check_repeat_obs=false,
                            alpha_observation=0.3,
                            alpha_action=0.3,);

    planner = POMDPs.solve(solver, pomdp);

    hr = HistoryRecorder();
    hist = simulate(hr, pomdp, planner)
    return hist;
end

function write_results(hist, dirname)
    serialized_tree = "$dirname/tree.jld2"
    cumulative_discounted_reward = discounted_reward(hist)

    for step in hist
        println("Action: $(step.a.id), Reward: $(step.r)")
    end
    println()
    println("Cumulative Discounted Reward: $cumulative_discounted_reward")

    info = first(ainfo_hist(hist))
    tree = D3Tree(info[:tree])
    
    @save serialized_tree tree;
end


dirname = get_date_dirname()
mkdir(dirname)

open("$dirname/reward.txt", "w") do io
    redirect_stdout(io) do
        @time hist = run_solver()
        write_results(hist, dirname);
    end
end