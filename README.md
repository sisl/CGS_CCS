# CCSPOMDPs
A toy pomdp model for Carbon Storage

Sample usage in REPL:
1. Activate with `] activate .`
2. Run a POMCPOW solver `include("scripts/solve.jl")`

## Adding Features
The POMDP formulation tracks suitability of various grid locations.
Suitability is an average of individual scores for each feature. If you have a feature that
you think would help determine suitability, you can add it with the following steps:

1. In `src/config.jl` add the name of your feature to `FEATURE_NAMES`
2. In `src/config.jl` add a prior belief of your variable's value `:featureName => (mean, variance)` to the `PRIOR_BELIEF` dictionary
3. In `src/config.jl` add action uncertainties to the `a_u` dictionary `(:action_type, :featureName) => variance`
    - This means that an action of `:action_type` can inform us of the value of `:featureName` with `variance`
4. Include logic on how your feature is to be scored in the `score_component` function, which returns a value 0 - 5

## Planning Links:

* [SharePoint](https://office365stanford-my.sharepoint.com/personal/ariefm_stanford_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fariefm%5Fstanford%5Fedu%2FDocuments%2FCGS%2DIntelligentData)
* [Planning + Action Items](https://docs.google.com/spreadsheets/d/1GVO2x4Y90s34S1VQMhuLiEPM-HxtHcT1YsQcQBFKgtk/edit?usp=sharing)
* [Progress Report](https://www.overleaf.com/7766495499gddrxnzyhmcg#ddcb23)