using Base: Generator
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
mutable struct BlackBox{S, F, A, K, P <: AbstractNLPModel}
  solver::S
  bb_function::F
  bb_args::Vector{A}
  bb_kwargs::Dict{Symbol, K}
  problems::Vector{Vector{P}}

  function BlackBox(
    solver::S,
    bb_function::F,
    bb_args::Union{Nothing, Vector{A}},
    bb_kwargs::Union{Nothing, Dict{Symbol, K}},
    problems::Vector{Vector{P}},
  ) where {S, F, A, K, P <: AbstractNLPModel}
    new{S, F, A, K, P}(solver, bb_function, bb_args, bb_kwargs, problems)
  end
end

function BlackBox(
  solver::S,
  bb_args::Vector{A},
  bb_kwargs::Dict{Symbol, K},
  problems::Vector{P},
) where {S, A, K, P <: AbstractNLPModel}
  problem_batches = create_batches(problems)
  return BlackBox(solver, default_black_box, bb_args, bb_kwargs, problem_batches)
end

function run_black_box(black_box::BlackBox{S,F,A,K,P}, new_param_values::AbstractVector{Float64}) where {S,F,A,K,P <:AbstractNLPModel}
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
  problem_batches::Vector{Vector{R}},
) where {P <: AbstractHyperParameter, R <: AbstractNLPModel}
  futures = Dict{Int, Future}()
  @sync for (worker_id, batch) in zip(workers(), problem_batches)
    @async futures[worker_id] = @spawnat worker_id let time = 0.0
      for nlp in batch
        bmark_result = @benchmark lbfgs($nlp, $solver_params) seconds = 10 samples = 5 evals = 1
        finalize(nlp)
        finalize(nlp)
        time += (median(bmark_result).time / 1.0e9)
      end
      return time
    end
  end
  solver_times = Dict{Int, Float64}()
  @sync for worker_id in workers()
    @async solver_times[worker_id] = fetch(futures[worker_id])
  end

  return [sum(values(solver_times))]
end

function create_batches(problems::Vector{P}) where {P <: AbstractNLPModel}
  nb_workers = length(workers())
  batches = [Vector{P}() for _ = 1:nb_workers]

  for (i, problem) in enumerate(problems)
    push!(batches[(i % nb_workers) + 1], problem)
  end
  return batches
end

# Surrogate defined by user 
# function default_black_box_surrogate(
#     solver_params::AbstractVector{P};
#     kwargs...,
# ) where {P<:AbstractHyperParameter}
#     max_time = 0.0
#     problems = CUTEst.select(; kwargs...)
#     for problem in problems
#         nlp = CUTEstModel(problem)
#         finalize(nlp)
#     end
#     return [max_time]
# end
