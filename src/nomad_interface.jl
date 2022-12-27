export ParameterOptimizationProblem, create_nomad_problem!, solve_with_nomad!

# TODO: add constraint support
"""
Struct that defines a problem that will be sent to NOMAD.jl.
"""
mutable struct ParameterOptimizationProblem{
  B <: AbstractBBModel,
  L <: AbstractLoadBalancer,
}
  nomad::Union{Nothing, NomadProblem}
  nlp::B
  c::Vector{Float64}
  load_balancer::L
  worker_data::Dict{Int, Vector{Vector{ProblemMetrics}}}
end

function ParameterOptimizationProblem(
  nlp::B;
  lb_choice::Symbol = :C,
) where {B <: AbstractBBModel}
  lb_choice ∈ (:G, :R, :C) || error("The load balancer option '$(lb_choice)' does not exist.")

  (lb_choice == :G) && (load_balancer = GreedyLoadBalancer(nlp.problems))
  (lb_choice == :R) && (load_balancer = RoundRobinLoadBalancer(nlp.problems))
  (lb_choice == :C) && (load_balancer = CombineLoadBalancer(nlp.problems))

  obj = ParameterOptimizationProblem(nlp, Vector{Float64}(), load_balancer)
  obj.c = format_constraints(obj)
  return obj
end

function ParameterOptimizationProblem(
  nlp::B,
  c::Vector{Float64},
  load_balancer::L,
) where {B <: AbstractBBModel, L <: AbstractLoadBalancer}
  worker_data = Dict{Int, Vector{Vector{ProblemMetrics}}}(
    i => Vector{Vector{ProblemMetrics}}() for i in workers()
  )
  ParameterOptimizationProblem(nothing, nlp, c, load_balancer, worker_data)
end

function create_nomad_problem!(
  param_opt_problem::ParameterOptimizationProblem{B, L};
  kwargs...,
) where {B <: AbstractBBModel, L <: AbstractLoadBalancer}
  # eval function:
  function eval_function(v::AbstractVector{Float64}; problem = param_opt_problem)
    eval_fct(v, problem)
  end
  bbmodel = param_opt_problem.nlp
  ncon = length(param_opt_problem.c)
  output_types = vcat(String["OBJ"], String["EB" for _ ∈ 1:ncon])
  nomad = NomadProblem(
    length(bbmodel.parameter_set),
    length(output_types),
    output_types,
    eval_function;
    input_types = input_types(bbmodel),
    granularity = granularities(values(bbmodel.parameter_set)),
    lower_bound = bbmodel.meta.lvar,
    upper_bound = bbmodel.meta.uvar,
  )
  set_nomad_options!(nomad.options; kwargs...)
  param_opt_problem.nomad = nomad
end

# define eval function here:
function eval_fct(
  v::Vector{Float64},
  param_opt_problem::ParameterOptimizationProblem{B, L},
) where {B <: AbstractBBModel, L <: AbstractLoadBalancer}
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
  param_opt_problem::ParameterOptimizationProblem{B, L},
  v::Vector{Float64},
) where {B <: AbstractBBModel, L <: AbstractLoadBalancer}
  set_values!(param_opt_problem.nlp.parameter_set, v)
  c = format_constraints(param_opt_problem)
  bb_output, new_worker_data = get_bb_output(param_opt_problem.nlp)
  update_worker_data!(param_opt_problem.worker_data, new_worker_data)
  update_lb!(param_opt_problem)
  return [bb_output, c...]
end

function update_lb!(
  param_opt_problem::ParameterOptimizationProblem{B, L},
) where {B <: AbstractBBModel, L <: AbstractLoadBalancer}
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
  p::ParameterOptimizationProblem{B, L},
) where {B <: AbstractBBModel, L <: AbstractLoadBalancer}
  @assert !isnothing(p.nlp) "error: Black Box not defined"
end

function solve_with_nomad!(
  problem::ParameterOptimizationProblem{B, L},
) where {B <: AbstractBBModel, L <: AbstractLoadBalancer}
  check_problem(problem)
  solve(problem.nomad, problem.nlp.meta.x0)
end

function input_types(nlp::AbstractBBModel)
  nomad_types = Vector{String}(undef, length(nlp.meta.x0))
  ifloat = nlp.bb_meta.ifloat
  iint = nlp.bb_meta.iint
  ibool = nlp.bb_meta.ibool
  for f_idx in ifloat
    nomad_types[f_idx] = "R"
  end
  for i_idx in iint
    nomad_types[i_idx] = "I"
  end
  for b_idx in ibool
    nomad_types[b_idx] = "B"
  end
  return nomad_types
end

granularities(x::S) where {S} = [granularity(xᵢ) for xᵢ in x]
granularity(::T) where {T <: Integer} = one(Float64)
granularity(::AbstractFloat) = Float64(0.0)

