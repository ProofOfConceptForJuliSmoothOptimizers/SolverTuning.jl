module SolverTuning

# stdlib.
using Pkg, LinearAlgebra, Distributed, Random

# JSO packages.
using BBModels, NLPModels

# Nomad.
using NOMAD
using NOMAD: NomadOptions

# Specific imports.
using ClusterManagers

worker_problems = Vector{Problem}()
# seed for random numbers:
Random.seed!(2017)

# include module components.
include("workers.jl")
include("load_balancer.jl")
include("run_bb_model.jl")
include("nomad_interface.jl")
include("constraints.jl")
include("solve.jl")
end
