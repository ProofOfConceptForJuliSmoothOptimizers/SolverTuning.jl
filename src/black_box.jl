"""
Black Box Object that will contain all the necessary info to execute the function with NOMAD:

solver: 
    should be an abstract solver

bb_function: 
    the function to be executed by NOMAD.
    This function should always take the vector of algorithmic parameters and the problem batches respectively in that order.
    i.e: 
    ```julia 
      function my_black_box(solver_params::AbstractVector{P}, problem_batches::Vector, your_args...; your_kwargs...)
          ...
          ...
          return [some_float64_value]
      end
    ```
    Note that this function must return a vector

bb_args:
    The other positional arguments of the black box.
    Note that if this vector is empty, the black box will still receive the solver parameters and the problem batches.
    Can be empty

bb_kwargs:
    The black box's keyword arguments. Can be empty.
"""
mutable struct BlackBox{S <: AbstractOptSolver, F<:Function, A, K}
  solver::S
  func::F
  args::Vector{A}
  kwargs::Dict{Symbol, K}

  function BlackBox(solver::S, func::F, args::Vector{A}, kwargs::Dict{Symbol, K}) where {S<: AbstractOptSolver, F<:Function, A, K}
    !isempty(args) || error("args must at least contain the solver parameters.")
    new{S,F,A,K}(solver, func, args, kwargs)
  end
end

function BlackBox(solver::S, func::F, kwargs::Dict{Symbol, K}) where {S<: AbstractOptSolver, F<:Function, K}
  args = [solver.parameters]
  return BlackBox(solver, func, args, kwargs)
end

function run_black_box(black_box::BlackBox{S,F,A,K}, new_param_values::AbstractVector{Float64}) where {S<: AbstractOptSolver,F<:Function,A,K}
  bb_func = black_box.func
  solver_params = black_box.solver.parameters
  args = black_box.args
  kwargs = black_box.kwargs
  [
    set_default!(param, param_value) for
    (param, param_value) in zip(solver_params, new_param_values)
  ]

  return bb_func(args...; kwargs...)
end

function eval_solver(solver_function::F, solver_params::AbstractVector{P}, args...; kwargs...) where {F<:Function, P<:AbstractHyperParameter}
  futures = Dict{Int64, Future}()
  @sync for worker_id in workers()
    @async futures[worker_id] = @spawnat worker_id let bmark_results=Dict{AbstractNLPModel, Trial}(), stats=Dict{AbstractNLPModel,AbstractExecutionStats}()
      global worker_problems
      for nlp in worker_problems
        bmark_result, stat = @benchmark_with_result $solver_function($nlp, $solver_params, $args...; $kwargs...) seconds = 10 samples = 5 evals = 1
        bmark_results[nlp] = bmark_result
        stats[nlp] = stat
        finalize(nlp)
      end
      return (bmark_results, stats)
    end
  end
  solver_results = Dict{Int, Tuple}()
  @sync for worker_id in workers()
    @async solver_results[worker_id] = fetch(futures[worker_id])
  end

  global workers_data
  worker_times = Dict(worker_id => 0.0 for worker_id in keys(solver_results))
  for (worker_id, solver_result) in solver_results
    bmark_trials, _ = solver_result
    worker_times[worker_id] = sum(median(trial).time/1.0e9 for trial in values(bmark_trials))
  end
  push!(workers_data, worker_times)

  bmark_results = merge([bmark_result for (bmark_result, _) ∈ values(solver_results)]...)
  stats_results = merge([stats_result for (_, stats_result) ∈ values(solver_results)]...)
  return bmark_results, stats_results
end

