export ParameterOptimizationProblem, create_nomad_problem!, solve_with_nomad!

"""
Struct that defines a problem that will be sent to NOMAD.jl.
"""
mutable struct ParameterOptimizationProblem{B <: BlackBox, L <: AbstractLoadBalancer}
  nomad::Union{Nothing, NomadProblem}
  black_box::B
  load_balancer::L
  nb_eval::Int
end

# TODO: Add Parametric type to solver (e.g Abstract Solver)
function ParameterOptimizationProblem(
  black_box::B,
  load_balancer::L,
) where {B <: BlackBox, L <: AbstractLoadBalancer}
  ParameterOptimizationProblem(nothing, black_box, load_balancer, 0)
end

function create_nomad_problem!(
  param_opt_problem::ParameterOptimizationProblem{B, L};
  kwargs...,
) where {B <: BlackBox, L <: AbstractLoadBalancer}
  # eval function:
  function eval_function(v::AbstractVector{Float64}; problem = param_opt_problem)
    eval_fct(v, problem)
  end

  solver_params = param_opt_problem.black_box.solver.parameters
  nomad = NomadProblem(
    length(solver_params),
    1,
    ["OBJ"],
    eval_function;
    input_types = input_types(solver_params),
    granularity = granularities(solver_params),
    lower_bound = lower_bounds(solver_params),
    upper_bound = upper_bounds(solver_params),
  )
  set_nomad_options!(nomad.options; kwargs...)
  param_opt_problem.nomad = nomad
end

# define eval function here: 
function eval_fct(
  v::AbstractVector{Float64},
  param_opt_problem::ParameterOptimizationProblem{B, L},
) where {B <: BlackBox, L <: AbstractLoadBalancer}
  lb = param_opt_problem.load_balancer
  success = false
  count_eval = false
  black_box_output = [Inf64]
  execute(lb, param_opt_problem.nb_eval)
  try
    black_box_output, bmark_results, stats_results = run_optim_problem(param_opt_problem, v)
    update_problems(lb, bmark_results)
    success = true
    count_eval = true
    param_opt_problem.nb_eval += 1
  catch exception
    println("exception occured while solving")
    showerror(stdout, exception)
    if isa(exception, CompositeException)
      showerror(stdout, exception.exceptions[1])
    end
  finally
    return success, count_eval, black_box_output
  end
end

function run_optim_problem(
  param_opt_problem::ParameterOptimizationProblem{B, L},
  new_param_values::AbstractVector{Float64},
) where {B <: BlackBox, L <: AbstractLoadBalancer}
  return run_black_box(param_opt_problem.black_box, new_param_values)
end

function set_nomad_options!(options::NomadOptions; kwargs...)
  for (field, value) in Dict(kwargs)
    setfield!(options, field, value)
  end
end

# Function that validates a parameter optimization problem
function check_problem(
  p::ParameterOptimizationProblem{B, L},
) where {B <: BlackBox, L <: AbstractLoadBalancer}
  @assert !isnothing(p.black_box) "error: Black Box not defined"
end

function solve_with_nomad!(
  problem::ParameterOptimizationProblem{B, L},
) where {B <: BlackBox, L <: AbstractLoadBalancer}
  check_problem(problem)
  solve(problem.nomad, current_param_values(problem.black_box.solver.parameters))
end
