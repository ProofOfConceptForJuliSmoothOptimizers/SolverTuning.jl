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
using NOMAD: NomadOptions

include("domains.jl")
include("parameters.jl")
include("lbfgs.jl")
include("nomad_interface.jl")

function default_black_box(
    solver_params::AbstractVector{P};
    kwargs...,
) where {P<:AbstractHyperParameter}
    max_time = 0.0
    problems = CUTEst.select(; kwargs...)
    for problem in problems
        nlp = CUTEstModel(problem)
        time_per_problem = @elapsed lbfgs(nlp, solver_params)
        finalize(nlp)
        max_time += time_per_problem
    end
    return [max_time]
end

function default_black_box_surrogate(
    solver_params::AbstractVector{P};
    kwargs...,
) where {P<:AbstractHyperParameter}
    max_time = 0.0
    n_problems = 10
    problems = CUTEst.select(; kwargs...)
    for i in rand(1:length(problems), n_problems)
        nlp = CUTEstModel(problems[i])
        result = lbfgs(nlp, solver_params)
        finalize(nlp)
        max_time += result.elapsed_time
    end
    return [max_time]
end

function main()
    nlp = ADNLPModel(
        x -> (x[1] - 1)^2 + 4 * (x[2] - 1)^2,
        zeros(2),
        name = "(x₁ - 1)² + 4(x₂ - 1)²",
    )
    # define params:
    mem = AlgorithmicParameter(5, IntegerRange(1, 100), "mem")
    τ₁ = AlgorithmicParameter(Float64(0.99), RealInterval(Float64(1.0e-4), 1.0), "τ₁")
    scaling = AlgorithmicParameter(true, BinaryRange(), "scaling")
    bk_max = AlgorithmicParameter(25, IntegerRange(10, 30), "bk_max")
    lbfgs_params = [mem, τ₁, scaling, bk_max]
    # define paramter tuning problem:
    solver = LBFGSSolver(nlp, lbfgs_params)
    # define problem suite
    param_optimization_problem = ParameterOptimizationProblem(
        solver,
        default_black_box,
        default_black_box_surrogate,
        true,
    )
    # CUTEst selection parameters
    bb_params = Dict(:min_var => 1, :max_var => 100, :max_con => 0, :only_free_var => true)
    # named arguments are options to pass to Nomad
    create_nomad_problem!(
        param_optimization_problem,
        bb_params;
        max_time = 300,
        display_unsuccessful = true,
    )
    result = solve_with_nomad!(param_optimization_problem)
    println(result)
end

main()
