export solve_bb_model

function solve_bb_model!(
  problem::ParameterOptimizationProblem{B, L},
) where {B <: AbstractBBModel, L <: AbstractLoadBalancer}
  check_problem(problem)
  return solve(problem.nomad, problem.nlp)
end

function solve_bb_model(bbmodel::AbstractBBModel; lb_choice = :C, kwargs...)
  param_optimization_problem = ParameterOptimizationProblem(bbmodel; lb_choice = lb_choice)
  let result = nothing, best_params = nothing
    try
      # named arguments are options to pass to Nomad
      create_nomad_problem!(param_optimization_problem; kwargs...)
      result = solve_with_nomad!(param_optimization_problem)
      result = result.x_best_feas
      param_names = Symbol[Symbol(i) for i in param_optimization_problem.nlp.bb_meta.x_n]
      best_params = (;zip(param_names, result)...)
      @info "Best feasible parameters: $best_params"
    catch e
      @error "Error occured while running NOMAD: $e"
      if isa(e, CompositeException)
        showerror(stdout, exception.exceptions[1])
      end
      best_params =
        (; zip(param_optimization_problem.nlp.bb_meta.x_n, values(param_optimization_problem.nlp.parameter_set))...)
    finally
      rmprocs(workers())
      return best_params, param_optimization_problem
    end
  end
end
