# Distributed computing
using ClusterManagers, Distributed

include("setup_workers.jl")

bb_kwargs = Dict(:min_var => 1, :max_var => 100, :max_con => 0, :only_free_var => true)

function decode_model(name::String)
    finalize(CUTEstModel(name))
end

function create_batches(nb_batches::Int64; kwargs...)
    # CUTEst problem selection and creation of problem batches:
    cutest_problems = CUTEst.select(;kwargs...)
    broadcast(decode_model, cutest_problems)
    batches = [[] for _ in 1:nb_sge_nodes]
    let idx = 0
        while length(cutest_problems) > 0
            problem = pop!(cutest_problems)
            push!(batches[(idx % nb_sge_nodes)  + 1], problem)
            idx += 1
        end
    end
    return batches
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

#create problem batches
batches = create_batches(nb_sge_nodes;bb_kwargs...)

# blackbox defined by user
function default_black_box(
    solver_params::AbstractVector{P};
    workers=workers(),
    problem_batches=batches,
    kwargs...,
) where {P<:AbstractHyperParameter}
    futures = Dict{Int, Future}()
    broadcast(b -> broadcast(decode_model, b), problem_batches)
    @sync for (worker_id, batch) in zip(workers, problem_batches)
        @async futures[worker_id] = @spawnat worker_id let time= 0.0
            for problem_name in batch
                nlp = CUTEstModel(problem_name; decode=false)
                bmark_result = @benchmark lbfgs($nlp, $solver_params) seconds=10 samples=5 evals=1
                finalize(nlp)
                finalize(nlp)
                time += (median(bmark_result).time/1.0e9)
            end
            return time
        end
    end
    solver_times = Dict{Int, Float64}()
    @sync for worker_id in workers
        @async solver_times[worker_id] = fetch(futures[worker_id])
    end

    return [sum(values(solver_times))]
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

# define problem suite
param_optimization_problem = ParameterOptimizationProblem(
    solver,
    default_black_box,
    default_black_box_surrogate,
    false,
)

# named arguments are options to pass to Nomad
create_nomad_problem!(
    param_optimization_problem,
    bb_kwargs;
    display_all_eval = true,
    max_time = 36000
)

# Execute Nomad
result = solve_with_nomad!(param_optimization_problem)
println(result)

rmprocs(workers())
    

