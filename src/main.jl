# stdlib
using LinearAlgebra, Logging, Printf

# JSO packages
using Krylov,
LinearOperators,
NLPModels,
NLPModelsModifiers,
SolverCore,
SolverTools,
ADNLPModels,
SolverTest,
CUTEst

using NOMAD
using NOMAD:NomadOptions

include("domains.jl")
include("parameters.jl")
include("lbfgs.jl")
include("nomad_interface.jl")

function main()
    nlp = ADNLPModel(
        x -> (x[1] - 1)^2 + 4 * (x[2] - 1)^2,
        zeros(2),
        name = "(x₁ - 1)² + 4(x₂ - 1)²",
    )
    # define params:
    mem = AlgorithmicParameter(2, IntegerRange(1, 100), "mem")
    τ₁ = AlgorithmicParameter(Float64(0.99), RealInterval(Float64(1.0e-4), 1.0), "τ₁")
    scaling = AlgorithmicParameter(true, BinaryRange(), "scaling")
    bk_max = AlgorithmicParameter(25, IntegerRange(10, 30), "bk_max")
    lbfgs_params = [mem, τ₁, scaling, bk_max]
    # define paramter tuning problem:
    solver = LBFGSSolver(nlp, lbfgs_params)
    # define problem suite
    param_optimization_problem = ParameterOptimizationProblem(solver, default_black_box, default_black_box_substitute, true; max_time=1800, display_unsuccessful=true)
    result = solve_with_nomad!(param_optimization_problem)
    println(result)
end

main()
