# State hyperparams
const NUM_LAYERS = 5
const NUM_LINES = 7
const NUM_WELLS = 15
const GRID_SIZE = 100

const SEISMIC_N_POINTS = 25

# Belief Initialization
PRIOR_BELIEF = Dict( # outputs shift, scale (variance)
    :permeability => (500, 300 * 300), # miniDarcy
    :topSealThickness => (45, 20 * 20) # meters
)

# Variogram hyperparams
const RANGE = 15.
const SILL = 3.0
const NUGGET = 0.1

# Action costs
const WELL_COST = 3
const SEISMIC_LINE_COST = 4

# Reward Constants
const SUITABILITY_THRESHOLD = 3.5
const SUITABILITY_BIAS = 0.7
const SUITABILITY_NSAMPLES = 50

# Action Uncertainty
a_u = Dict(
    # verify variance vs std
    (:well_action, :z) => 0.1, # within 3 m
    (:well_action, :permeability) => 0.1,
    (:well_action, :topSealThickness) => 0.1,

    (:seismic_action, :z) => 0.5,
    (:seismic_action, :permeability) => 0.5, # 
    (:seismic_action, :topSealThickness) => 0.5,
)

ACTION_UNCERTAINTY = DefaultDict(0.1, a_u)