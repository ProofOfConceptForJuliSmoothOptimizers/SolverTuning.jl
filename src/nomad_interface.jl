export ParameterOptimizationProblem, create_nomad_problem!, solve_with_nomad!

# TODO: add constraint support
"""
Struct that defines a problem that will be sent to NOMAD.jl.
"""
mutable struct ParameterOptimizationProblem{T, S, B <: AbstractBBModel{T, S}, L <: AbstractLoadBalancer}
  nomad::Union{Nothing, NomadProblem}
  nlp::B
  x::S
  load_balancer::L
end

function ParameterOptimizationProblem(nlp::B; is_load_balanced=true) where {T, S, B <: AbstractBBModel{T, S}}
  load_balancer = is_load_balanced ? GreedyLoadBalancer(nlp.problems) : RoundRobinLoadBalancer(nlp.problems)
  ParameterOptimizationProblem(nlp, deepcopy(nlp.x0), load_balancer)
end

function ParameterOptimizationProblem(
  nlp::B,
  x::S,
  load_balancer::L,
) where {T, S, B <: AbstractBBModel{T, S}, L <: AbstractLoadBalancer}
  ParameterOptimizationProblem(nothing, nlp, x, load_balancer)
end

function create_nomad_problem!(
  param_opt_problem::ParameterOptimizationProblem{T, S, B, L};
  kwargs...,
) where {T, S, B <: AbstractBBModel{T, S}, L <: AbstractLoadBalancer}
  # eval function:
  function eval_function(v::AbstractVector{Float64}; problem = param_opt_problem)
    eval_fct(v, problem)
  end

  solver_params = param_opt_problem.x
  bbmodel = param_opt_problem.nlp
  output_types = ["EB" for _ in 1:bbmodel.meta.ncon]
  pushfirst!(output_types, "OBJ")
  nomad = NomadProblem(
    length(solver_params),
    length(output_types),
    output_types,
    eval_function;
    input_types = input_types(solver_params),
    granularity = granularities(solver_params),
    lower_bound = lower_bounds(bbmodel),
    upper_bound = upper_bounds(bbmodel),
  )
  set_nomad_options!(nomad.options; kwargs...)
  param_opt_problem.nomad = nomad
end

# define eval function here: 
function eval_fct(
  v::Vector{Float64},
  param_opt_problem::ParameterOptimizationProblem{T, S, B, L},
) where {T, S, B <: AbstractBBModel{T, S}, L <: AbstractLoadBalancer}
  lb = param_opt_problem.load_balancer
  success = false
  count_eval = false
  black_box_output = [Inf64]
  execute(lb)
  try
    black_box_output = run_optim_problem(param_opt_problem, v)
    success = true
    count_eval = true
  catch exception
    @error "Exception occured while solving"
    showerror(stdout, exception)
    if isa(exception, CompositeException)
      showerror(stdout, exception.exceptions[1])
    end
  finally
    return success, count_eval, black_box_output
  end
end

# Evaluate constraints here:
function run_optim_problem(
  param_opt_problem::ParameterOptimizationProblem{T, S, B, L},
  v::Vector{Float64},
) where {T, S, B <: AbstractBBModel{T, S}, L <: AbstractLoadBalancer}
  update!(param_opt_problem, v)
  return [run_bb_model(param_opt_problem.nlp, param_opt_problem.x)]
end

function update!(
  param_opt_problem::ParameterOptimizationProblem{T, S, B, L},
  v::Vector{Float64}
) where {T, S, B <: AbstractBBModel{T, S}, L <: AbstractLoadBalancer}
  for (xᵢ,vᵢ) in zip(param_opt_problem.x, v)
    if nomad_type(xᵢ) == "B"
      xᵢ = Bool(round(param_type, vᵢ))
    elseif nomad_type(xᵢ) == "I"
      xᵢ = round(param_type, vᵢ)
    else
      param_type = typeof(xᵢ)
      xᵢ = param_type(vᵢ)
    end
  end
end

function set_nomad_options!(options::NomadOptions; kwargs...)
  for (field, value) in Dict(kwargs)
    setfield!(options, field, value)
  end
end

# Function that validates a parameter optimization problem
function check_problem(
  p::ParameterOptimizationProblem{T, S, B, L},
) where {T, S, B <: AbstractBBModel, L <: AbstractLoadBalancer}
  @assert !isnothing(p.black_box) "error: Black Box not defined"
end

function solve_with_nomad!(
  problem::ParameterOptimizationProblem{T, S, B, L},
) where {T, S, B <: AbstractBBModel, L <: AbstractLoadBalancer}
  check_problem(problem)
  solve(problem.nomad, current_param_values(problem.black_box.solver_params))
end

input_types(x::S) where S = [nomad_type(xᵢ) for xᵢ in x]


nomad_type(::Type{T}) where {T <: Real} = "R"
nomad_type(::Type{T}) where {T <: Integer} = "I"
nomad_type(::Type{T}) where {T <: Bool} = "B"


granularities(x::S) where S = [granularity(xᵢ) for xᵢ in x]
granularity(::T) where {T <: Union{Bool, Int}} = one(Float64)
granularity(::Float64) = Float64(1.0e-5)
granularity(::Float32) = Float64(1.0e-4)
granularity(::Float16) = Float64(1.0e-3)

function lower_bounds(nlp::BBModel)
  lvar = nlp.meta.lvar
  return [Float64(l) for l in lvar]
end

function upper_bounds(nlp::BBModel)
  uvar = nlp.meta.uvar
  return [Float64(u) for u in uvar]
end

