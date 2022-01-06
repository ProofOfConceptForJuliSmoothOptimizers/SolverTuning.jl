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
mutable struct BlackBox{S, F, A, K, P}
  solver::S
  bb_function::F
  bb_args::Vector{A}
  bb_kwargs::Dict{Symbol, K}
  problems::Dict{Symbol, P}

  function BlackBox(
    solver::S,
    bb_function::F,
    bb_args::Union{Nothing, Vector{A}},
    bb_kwargs::Union{Nothing, Dict{Symbol, K}},
    problems::Dict{Symbol, P},
  ) where {S, F, A, K, P}
    new{S, F, A, K, P}(solver, bb_function, bb_args, bb_kwargs, problems)
  end
end

function BlackBox(
  solver::S,
  bb_args::Vector{A},
  bb_kwargs::Dict{Symbol, K},
  problems::Dict{Symbol, P},
) where {S, A, K, P}
  set_worker_problems(problems)
  return BlackBox(solver, default_black_box, bb_args, bb_kwargs, problems)
end

function run_black_box(black_box::BlackBox{S,F,A,K,P}, new_param_values::AbstractVector{Float64}) where {S,F,A,K,P}
  bb_func = black_box.bb_function
  solver_params = black_box.solver.parameters
  problems = black_box.problems
  args = black_box.bb_args
  kwargs = black_box.bb_kwargs
  [
    set_default!(param, param_value) for
    (param, param_value) in zip(solver_params, new_param_values)
  ]

  return bb_func(solver_params, problems, args...; kwargs...)
end

function default_black_box(
  solver_params::AbstractVector{P},
  problems::Dict{Symbol, R},
) where {P <: AbstractHyperParameter, R}
  futures = Dict{Int, Future}()
  @sync for worker_id in workers()
    @async futures[worker_id] = @spawnat worker_id let time = 0.0, stats=nothing
      global worker_problems
      for p in worker_problems
        nlp = get_problem(p)
        bmark_result, stats = @benchmark_with_result lbfgs($nlp, $solver_params) seconds = 10 samples = 5 evals = 1
        problems[p] = stats
        finalize(nlp)
        finalize(nlp)
        time += (median(bmark_result).time / 1.0e9)
      end
      return time, stats
    end
  end
  solver_times = Dict{Int, Tuple}()
  @sync for worker_id in workers()
    @async solver_times[worker_id] = fetch(futures[worker_id])
  end

  return [sum(time for (time, stats) in values(solver_times))]
end

function set_worker_problems(problems::Dict{Symbol, P}) where {P}
  problem_def_future = Future[]
  problem_names = keys(problems)
    for worker_id in Iterators.cycle(workers())
        try
            problem, problem_names = Iterators.peel(problem_names)
            push!(problem_def_future, remotecall(add_worker_problem, worker_id, problem))
        catch exception
          !isa(exception, BoundsError) || break
        end
    end
    @sync for problem_future in problem_def_future
      @async fetch(problem_future)
    end
end
