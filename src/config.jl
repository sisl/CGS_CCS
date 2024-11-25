# State hyperparams
const NUM_LAYERS = 5
const NUM_LINES = 7
const NUM_WELLS = 15
const GRID_SIZE = 100
const SPACING = 100 # 100 m grid cells

const SEISMIC_N_POINTS = 15

@enum RockType SANDSTONE=1 SILTSTONE=2 SHALE=3

# Belief Initialization
PRIOR_BELIEF = Dict( # outputs shift, scale (variance)
    # For permeability, log transform: Actual Sandstone range is 10 - 200,
    # ln range is 2.3 - 5.3, mean is 3.8, std is 1.5
    # according to https://www.saltworkconsultants.com/carbonate-porosity-sandstone-vs-carbonate/
    (:permeability, SANDSTONE) => (3.8, 1.5 * 1.5), # log(miniDarcy) 
    # Siltstone: 0.04 mD to 0.8 mD, log range is -3.22 to -0.22, mean is -1.72, std is 1.5
    # https://www.researchgate.net/figure/Cross-plot-of-horizontal-and-vertical-permeability-and-axial-and-diametral-pointload_fig9_278029544
    (:permeability, SILTSTONE) => (-1.72, 1.5 * 1.5), # log(miniDarcy)
    # Shale: 0.001 mD to 1 mD, log range is -6.9 to 0, mean is -3.45, std is 1.5
    # https://www.sciencedirect.com/science/article/pii/S0016236114009429#:~:text=%E2%80%A2,10%E2%88%927%20and%201.2%20mD.
    (:permeability, SHALE) => (-3.45, 1.5 * 1.5), # log(miniDarcy)

    # The topseal is independent of rock type (Any rock type can have any type of seal above it)
    (:topSealThickness, SANDSTONE) => (45, 15 * 15), # meters
    (:topSealThickness, SILTSTONE) => (45, 15 * 15), # meters
    (:topSealThickness, SHALE) => (45, 15 * 15), # meters

)

# Action costs
const WELL_COST = 3
const SEISMIC_LINE_COST = 4

# Reward Constants
const SUITABILITY_THRESHOLD = 3.5
const SUITABILITY_CONF_THRESHOLD = 0.8
const SUITABILITY_BIAS = 0.7
const SUITABILITY_NSAMPLES = 25
const λ_1 = 10
const λ_2 = 1e-4

# Action Uncertainty
# Get references for these

# Rock types: sandstone, siltstone (greywacke), shale
# Update permeability conditioned on rock type

# Use Belief MCTS as a baseline policy. If that doesn't work then POMCPOW ig.
# Does this beat random policy, grid/max coverage policy
# If far wells, include those, otherwise ensure coverage (grid) with budget

# Sampling plans in optimization textbook
a_u = Dict(
    (:well_action, :z) => 9, # within 3 m
    (:well_action, :permeability) => .05 * 0.05, # miniDarcy log transform
    (:well_action, :topSealThickness) => 4, # std 2 m

    (:seismic_action, :z) => 100, # within 10 m
    (:seismic_action, :topSealThickness) => 400, # within 20 m
)

ACTION_UNCERTAINTY = DefaultDict(-1., a_u)

# Variogram hyperparams
# Variogram hyperparams for permeability, for each rock type
# Rock type ordering is same as enum, Sandstone, Siltstone, Shale
const PERMEABILITY_NUGGET = 0.1, 0.05, 0.05 # (log(mD))
const PERMEABILITY_SILL = 2.0, 0.5, 0.5 # (log(mD))
const PERMEABILITY_RANGE = 1000, 500, 300 # meters

# Variogram hyperparams for topSealThickness, for each rock type
# Note all the values are the same because the topseal is independent of rock type
const TOPSEAL_NUGGET = 2, 2, 2 # meters
const TOPSEAL_SILL = 500, 500, 500 # meters
const TOPSEAL_RANGE = 2000, 2000, 2000 # meters

# Variogram hyperparams for z, for each rock type
const Z_NUGGET = 2, 2, 2 # meters
const Z_SILL = 50, 50, 50 # meters
const Z_RANGE = 1500, 1500, 1500 # meters

VARIOGRAM_HYPERPARAMS = Dict(
    :permeability => (PERMEABILITY_NUGGET, PERMEABILITY_SILL, PERMEABILITY_RANGE),
    :topSealThickness => (TOPSEAL_NUGGET, TOPSEAL_SILL, TOPSEAL_RANGE),
    :z => (Z_NUGGET, Z_SILL, Z_RANGE)
)