# stdlib
using LinearAlgebra, Logging, Printf, DataFrames

# Distributed computing
using ClusterManagers, Distributed

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

# setup julia workers on SGE:
nb_sge_nodes = 24
worker_pids = addprocs_sge(nb_sge_nodes; qsub_flags=`-q hs22 -V`, exeflags="--project=.", wd=joinpath(ENV["HOME"], "julia_worker_logs"))

# Define functions on all workers
@everywhere let packages = ["Krylov","LinearOperators","NLPModels","NLPModelsModifiers","SolverCore",
    "SolverTools","ADNLPModels","SolverTest","CUTEst"]
    using Pkg
    Pkg.add(packages)
    using Distributed
    Krylov,
    LinearOperators,
    NLPModels,
    NLPModelsModifiers,
    SolverCore,
    SolverTools,
    ADNLPModels,
    SolverTest,
    CUTEst,
    BenchmarkTools
end

@everywhere function evalmodels(problem_names::Vector{String}, solver_params::Vector{P}) where {P<:AbstractHyperParameter}
    time = 0.0
    for problem_name in problem_names
        nlp = CUTEstModel(problem_name; decode=false)
        benchmark = @benchmarkable lbfgs($nlp, $solver_params) seconds=300 samples=5 evals=1
        result = run(benchmark)
        finalize(nlp)
        # result is given in ns. Converting to seconds:
        time += (median(result).time/1.0e9)
    end
    return time
end

# blackbox defined by user
function default_black_box(
    solver_params::AbstractVector{P};
    workers=worker_pids,
    problem_batches=batches,
    kwargs...,
) where {P<:AbstractHyperParameter}
    total_time = 0.0
    futures = Future[]
    for (worker_id, batch) in zip(workers, problem_batches)
        future_time = @spawnat worker_id evalmodels(batch, solver_params)
        push!(futures, future_time)
    end
    @sync for time_future in futures
        @async begin
            solver_time = fetch(time_future)
            total_time += solver_time
        end
    end

    return [total_time]
end

# Surrogate defined by user 
function default_black_box_surrogate(
    solver_params::AbstractVector{P};
    kwargs...,
) where {P<:AbstractHyperParameter}
    max_time = 0.0
    problems = CUTEst.select(; kwargs...)
    for problem in problems
        nlp = CUTEstModel(problem)
        finalize(nlp)
    end
    return [max_time]
end

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
initial_lbfgs_params = deepcopy(lbfgs_params)

# define paramter tuning problem:
solver = LBFGSSolver(nlp, lbfgs_params)

# define problem suite
param_optimization_problem = ParameterOptimizationProblem(
    solver,
    default_black_box,
    default_black_box_surrogate,
    false,
)

# CUTEst problem selection and creation of problem batches:
bb_kwargs = Dict(:min_var => 1, :max_var => 100, :max_con => 0, :only_free_var => true)
cutest_problems = CUTEst.select(;bb_kwargs...)
broadcast(p -> finalize(CUTEstModel(p)), cutest_problems)
batches = [[] for i in 1:nb_sge_nodes]
let idx = 0
    while length(cutest_problems) > 0
        problem = pop!(cutest_problems)
        push!(batches[(idx % n_workers)  + 1], problem)
        idx += 1
    end
end
broadcast(x -> println(length(x)), batches)

# named arguments are options to pass to Nomad
create_nomad_problem!(
    param_optimization_problem,
    bb_kwargs;
    display_all_eval = true,
    max_time = 300,
)

# Execute Nomad
result = solve_with_nomad!(param_optimization_problem)
println(result)

# SolverBenchmark stats:
problems = (CUTEstModel(p) for p in CUTEst.select(;bb_kwargs...))
solvers = Dict(
  :lbfgs_old => model -> lbfgs(model, initial_lbfgs_params),
  :lbfgs_new => model -> lbfgs(model, param_optimization_problem.solver.parameters)
)
stats = bmark_solvers(solvers, problems)

open(joinpath(ENV["HOME"],"problem_stats.md"), "w") do io
    pretty_stats(io, stats)
end
    

