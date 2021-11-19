# stdlib
using LinearAlgebra, Logging, Printf, DataFrames

# JSO packages
using Krylov,
    LinearOperators,
    NLPModels,
    NLPModelsModifiers,
    SolverCore,
    SolverTools,
    ADNLPModels,
    SolverTest,
    SolverBenchmark,
    CUTEst

using BenchmarkTools
using NOMAD
using NOMAD: NomadOptions

include("domains.jl")
include("parameters.jl")
include("lbfgs.jl")
include("nomad_interface.jl")

# important for stats creation 
# id = 1
# Define SolverBenchmark data:
# benchmark_stats = Dict{Symbol, Any}(:lbfgs => DataFrame(id=Int64[], name=String[], status=Symbol[], f=Float64[], t=Float64[], iter=Int64[]))

function default_black_box(
    solver_params::AbstractVector{P};
    stats = benchmark_stats[:lbfgs],
    kwargs...,
) where {P<:AbstractHyperParameter}
    max_time = 0.0
    problems = CUTEst.select(; kwargs...)
    for problem in problems
        global id
        nlp = CUTEstModel(problem)
        result = lbfgs(nlp, solver_params)
        finalize(nlp)
        push!(stats, [id, problem, result.status, result.objective, result.elapsed_time, result.iter])
        max_time += result.elapsed_time
        id += 1
    end
    return [max_time]
end

function default_black_box_surrogate(
    solver_params::AbstractVector{P};
    # stats = benchmark_stats[:lbfgs],
    kwargs...,
) where {P<:AbstractHyperParameter}
    max_time = 0.0
    n_problems = 10
    problems = CUTEst.select(; kwargs...)
    for i in rand(1:length(problems), n_problems)
        # global id
        nlp = CUTEstModel(problems[i])
        result = @benchmark lbfgs($nlp, $solver_params) samples=40 evals=1
        finalize(nlp)
        # push!(stats, [id, problems[i], result.status, result.objective, result.elapsed_time, result.iter])
        # result is given in ns. Converting to seconds:
        max_time += (median(result).time/1.0e9)
        # id += 1
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
    bb_kwargs = Dict(:min_var => 1, :max_var => 100, :max_con => 0, :only_free_var => true)
    # named arguments are options to pass to Nomad
    create_nomad_problem!(
        param_optimization_problem,
        bb_kwargs;
        display_all_eval = true,
    )
    # Execute Nomad
    result = solve_with_nomad!(param_optimization_problem)
    println(result)
    pretty_stats(benchmark_stats[:lbfgs])
end

main()
