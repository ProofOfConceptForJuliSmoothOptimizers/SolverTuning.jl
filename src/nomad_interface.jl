export ParameterOptimizationProblem, create_nomad_problem!, solve_with_nomad!

# TODO: add constraint support
"""
Struct that defines a problem that will be sent to NOMAD.jl.
"""
mutable struct ParameterOptimizationProblem{
  T,
  S,
  B <: AbstractBBModel{T, S},
  L <: AbstractLoadBalancer,
}
  nomad::Union{Nothing, NomadProblem}
  nlp::B
  x::S
  c::Vector{Float64}
  load_balancer::L
  worker_data::Dict{Int, Vector{Vector{ProblemMetrics}}}
end

function ParameterOptimizationProblem(
  nlp::B;
  lb_choice::Symbol = :C,
) where {T, S, B <: AbstractBBModel{T, S}}
  load_balancer = CombineLoadBalancer(nlp.problems)
  if lb_choice == :G
    load_balancer = GreedyLoadBalancer(nlp.problems)
  elseif lb_choice == :R
    load_balancer = RoundRobinLoadBalancer(nlp.problems)
  end
  lb_choice ∈ (:G, :R, :C) ||
    @warn "this load balancer option does not exist. Choosing the Combine algorithm."
  obj = ParameterOptimizationProblem(nlp, deepcopy(nlp.meta.x0), Vector{Float64}(), load_balancer)
  obj.c = format_constraints(obj)
  return obj
end

function ParameterOptimizationProblem(
  nlp::B,
  x::S,
  c::Vector{Float64},
  load_balancer::L,
) where {T, S, B <: AbstractBBModel{T, S}, L <: AbstractLoadBalancer}
  worker_data = Dict{Int, Vector{Vector{ProblemMetrics}}}(
    i => Vector{Vector{ProblemMetrics}}() for i in workers()
  )
  ParameterOptimizationProblem(nothing, nlp, x, c, load_balancer, worker_data)
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
  ncon = length(param_opt_problem.c)
  output_types = ["EB" for _ = 1:ncon]
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
  n_con = length(param_opt_problem.c)
  success = false
  count_eval = false
  black_box_output = [Inf64]
  push!(black_box_output, [0.0 for _ = 1:n_con]...)
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
  c = format_constraints(param_opt_problem)
  # @info "new params: $(param_opt_problem.x)"
  # @info "constraints values: $c"
  bb_output, new_worker_data = run_bb_model(param_opt_problem.nlp, param_opt_problem.x)
  update_worker_data!(param_opt_problem.worker_data, new_worker_data)
  update_lb!(param_opt_problem)
  return [bb_output, c...]
end

function update!(
  param_opt_problem::ParameterOptimizationProblem{T, S, B, L},
  v::Vector{Float64},
) where {T, S, B <: AbstractBBModel{T, S}, L <: AbstractLoadBalancer}
  for (i, vᵢ) in zip(1:length(param_opt_problem.x), v)
    if nomad_type(param_opt_problem.x[i]) == "B"
      param_opt_problem.x[i] = Bool(round(Int, vᵢ))
    elseif nomad_type(param_opt_problem.x[i]) == "I"
      param_opt_problem.x[i] = round(Int, vᵢ)
    else
      param_type = typeof(param_opt_problem.x[i])
      param_opt_problem.x[i] = param_type(vᵢ)
    end
  end
end

function update_lb!(
  param_opt_problem::ParameterOptimizationProblem{T, S, B, L},
) where {T, S, B <: AbstractBBModel{T, S}, L <: AbstractLoadBalancer}
  worker_data = param_opt_problem.worker_data
  lb = param_opt_problem.load_balancer
  for (_, bb_iterations) in worker_data
    last_iteration = last(bb_iterations)
    for p_metric in last_iteration
      new_time = median(get_times(p_metric))
      lb.problems[get_pb_id(p_metric)].weight += new_time
    end
  end
end

function update_worker_data!(
  data::Dict{Int, Vector{Vector{ProblemMetrics}}},
  new_data::Dict{Int, Vector{ProblemMetrics}},
)
  for w_id in workers()
    push!(data[w_id], new_data[w_id])
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
  @assert !isnothing(p.nlp) "error: Black Box not defined"
end

function solve_with_nomad!(
  problem::ParameterOptimizationProblem{T, S, B, L},
) where {T, S, B <: AbstractBBModel, L <: AbstractLoadBalancer}
  check_problem(problem)
  solve(problem.nomad, [Float64(xᵢ) for xᵢ in problem.x])
end

input_types(x::S) where {S} = [nomad_type(xᵢ) for xᵢ in x]

nomad_type(::T) where {T <: Real} = "R"
nomad_type(::T) where {T <: Integer} = "I"
nomad_type(::T) where {T <: Bool} = "B"

granularities(x::S) where {S} = [granularity(xᵢ) for xᵢ in x]
granularity(::T) where {T <: Union{Bool, Int}} = one(Float64)
granularity(::Float64) = Float64(0.0)
granularity(::Float32) = eps(Float64)
granularity(::Float16) = Float64(1.0e-3)

function lower_bounds(nlp::BBModel)
  lvar = nlp.meta.lvar
  return [Float64(l) for l in lvar]
end

function upper_bounds(nlp::BBModel)
  uvar = nlp.meta.uvar
  return [Float64(u) for u in uvar]
end
